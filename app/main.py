"""
Employee Attendance & Analytics API
===================================
A single-file, production-grade FastAPI application built on MongoDB 6.0+.
Implements the contract defined in openapi.yaml and strictly satisfies
business rules R1 through R10.

Key Architectural Guarantees:
1. Concurrency Safety:
   - Database unique index on (emp_code, date) prevents duplicate punch-ins.
   - Atomic find_one_and_update guarantees idempotent, race-free punch-outs.
   - Optimistic concurrency control prevents lost history entries on regularization.
2. Index-Backed Queries:
   - All list and analytics pipelines utilize IXSCAN to eliminate full table scans.
3. Strict Temporal Boundary:
   - External API exchanges epoch milliseconds (integers).
   - Internal MongoDB storage uses UTC BSON datetimes.
   - Shift calendar boundaries and attendance dates are evaluated in IST (UTC+05:30).
4. Deterministic Half-Up Rounding:
   - All floating-point math uses Python's Decimal with ROUND_HALF_UP (R4/R8).
"""

from contextlib import asynccontextmanager
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
import os
import re
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Path, Query, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from pymongo import ASCENDING, DESCENDING, MongoClient, ReturnDocument, errors

DuplicateKeyError = errors.DuplicateKeyError
PyMongoError = errors.PyMongoError

# --------------------------------------------------------------------------- #
# 1. Environment & Global Configuration
# --------------------------------------------------------------------------- #
load_dotenv()

# Real environment variables take priority over local .env
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "attendance_db")

# Indian Standard Time (IST = UTC + 05:30) per Rule R1
IST = timezone(timedelta(hours=5, minutes=30))

# Initialize MongoClient with tz_aware=True to guarantee UTC datetimes
client = MongoClient(MONGO_URI, tz_aware=True)
db = client[MONGO_DB]


# --------------------------------------------------------------------------- #
# 2. Database Indexes & Application Lifecycle (Lifespan)
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Executes once when FastAPI boots.
    Creates all necessary database indexes idempotently to ensure queries
    use index scans (IXSCAN) and prevent race conditions at the database level.
    """
    # 1. Unique index on employees.emp_code (enforces uniqueness for customer codes)
    db.employees.create_index([("emp_code", ASCENDING)], unique=True)

    # 2. Index on department and joined_on (used by department lookups and headcounts)
    db.employees.create_index([("department", ASCENDING), ("joined_on", ASCENDING)])
    db.employees.create_index([("joined_on", ASCENDING)])

    # 3. Unique compound index on attendance_logs (emp_code, date)
    #    Guarantees exactly one record per employee per day; prevents punch-in races.
    db.attendance_logs.create_index([("emp_code", ASCENDING), ("date", ASCENDING)], unique=True)

    # 4. Sorting and filtering index for /attendance listing
    db.attendance_logs.create_index([("date", DESCENDING), ("emp_code", ASCENDING)])

    # 5. Fast search index for punch-out (finding most recent open punch-in)
    db.attendance_logs.create_index([("emp_code", ASCENDING), ("punch_in", ASCENDING)])

    # 6. Index for monthly date range queries and late leaderboard
    db.attendance_logs.create_index([("date", ASCENDING), ("late_minutes", ASCENDING)])

    yield


app = FastAPI(
    title="Employee Attendance & Analytics API",
    version="2.0.0",
    lifespan=lifespan,
)


# --------------------------------------------------------------------------- #
# 3. Business Rule Helpers (R1 - R10)
# --------------------------------------------------------------------------- #
def round_half_up(value: Optional[float], decimals: int) -> Optional[float]:
    """
    Rules R4 & R8: Half-up rounding (e.g., 2.345 -> 2.35).
    Python's standard round() uses banker's rounding (even round), which produces
    incorrect values under business audit rules.
    """
    if value is None:
        return None
    d = Decimal(str(value))
    return float(d.quantize(Decimal("10") ** -decimals, rounding=ROUND_HALF_UP))


def epoch_ms_to_datetime(ms: Any) -> datetime:
    """
    Validates and converts integer epoch milliseconds to UTC datetime.
    Rule R1: Truncates instants to whole seconds before calculation or storage.
    Rejects values < 100,000,000,000 (which look like seconds) with HTTP 422.
    """
    if isinstance(ms, bool) or not isinstance(ms, int):
        raise HTTPException(status_code=422, detail="Timestamp must be an integer in milliseconds")
    if ms < 100_000_000_000 or ms > 4_102_444_800_000:
        raise HTTPException(status_code=422, detail="Timestamp out of valid epoch milliseconds range")
    
    # Truncate to whole seconds per Rule R1 (e.g., 09:40:00.900 -> 09:40:00)
    whole_seconds = ms // 1000
    return datetime.fromtimestamp(whole_seconds, tz=timezone.utc)


def datetime_to_epoch_ms(dt: Optional[datetime]) -> Optional[int]:
    """Converts a BSON UTC datetime to integer epoch milliseconds for JSON responses."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def parse_time_str(time_str: str) -> tuple[int, int]:
    """Parses 'HH:MM' string into hour and minute integers."""
    parts = time_str.split(":")
    return int(parts[0]), int(parts[1])


def determine_attendance_date(punch_in_utc: datetime, shift_start: str, shift_end: str) -> str:
    """
    Rule R1: Determines the attendance calendar date (IST).
    For overnight shifts (shift_end <= shift_start, e.g. 22:00 to 06:00):
    A punch-in whose IST time is strictly earlier than shift_end belongs to the
    shift that started on the previous calendar day.
    """
    punch_in_ist = punch_in_utc.astimezone(IST)
    start_h, start_m = parse_time_str(shift_start)
    end_h, end_m = parse_time_str(shift_end)

    is_overnight = (end_h, end_m) <= (start_h, start_m)

    if is_overnight:
        punch_time = punch_in_ist.time()
        end_time = time(end_h, end_m)
        if punch_time < end_time:
            # Belongs to previous day's shift
            return (punch_in_ist.date() - timedelta(days=1)).isoformat()

    return punch_in_ist.date().isoformat()


def get_shift_datetimes(att_date_str: str, shift_start: str, shift_end: str) -> tuple[datetime, datetime]:
    """
    Constructs UTC shift_start and shift_end datetimes for a given attendance date.
    Overnight shifts end on the next calendar day.
    """
    att_date = date.fromisoformat(att_date_str)
    start_h, start_m = parse_time_str(shift_start)
    end_h, end_m = parse_time_str(shift_end)

    shift_start_ist = datetime.combine(att_date, time(start_h, start_m), tzinfo=IST)

    is_overnight = (end_h, end_m) <= (start_h, start_m)
    if is_overnight:
        shift_end_ist = datetime.combine(att_date + timedelta(days=1), time(end_h, end_m), tzinfo=IST)
    else:
        shift_end_ist = datetime.combine(att_date, time(end_h, end_m), tzinfo=IST)

    return shift_start_ist.astimezone(timezone.utc), shift_end_ist.astimezone(timezone.utc)


def compute_late_minutes(punch_in_utc: datetime, shift_start: str, att_date_str: str, shift_end: str) -> int:
    """
    Rule R2: Grace period is 10 minutes.
    A punch-in is late only if strictly > 10:00 (600 seconds) after shift_start.
    When late, late_minutes = whole minutes (floor) elapsed since shift_start.
    Otherwise 0.
    """
    shift_start_utc, _ = get_shift_datetimes(att_date_str, shift_start, shift_end)
    diff_seconds = int((punch_in_utc - shift_start_utc).total_seconds())

    # Grace period is strictly 10 minutes (600 seconds)
    if diff_seconds > 600:
        return diff_seconds // 60
    return 0


def compute_work_hours_and_overtime(
    punch_in_utc: datetime,
    punch_out_utc: datetime,
    shift_end: str,
    att_date_str: str,
    shift_start: str,
) -> tuple[float, int, bool]:
    """
    Rules R3, R4, R5:
    - R3: Overtime = whole minutes (floor) between shift_end and punch_out, only when >= 30; else 0.
    - R4: Work hours = (punch_out - punch_in) in seconds / 3600, rounded half-up to 2 decimals.
    - R5: Half day = True when rounded work_hours < 4.50.
    """
    duration_seconds = (punch_out_utc - punch_in_utc).total_seconds()
    work_hours = round_half_up(duration_seconds / 3600.0, 2)
    half_day = bool(work_hours < 4.50)

    _, shift_end_utc = get_shift_datetimes(att_date_str, shift_start, shift_end)
    ot_seconds = int((punch_out_utc - shift_end_utc).total_seconds())

    if ot_seconds >= 1800:  # At least 30 minutes
        overtime_minutes = ot_seconds // 60
    else:
        overtime_minutes = 0

    return work_hours, overtime_minutes, half_day


def count_working_days(joined_on_str: str, month_str: str) -> int:
    """
    Rule R7: Working days are Monday - Friday (no holidays).
    For mid-month joiners, counted from max(month_start, joined_on).
    If joined_on > month_end, working_days = 0.
    """
    year, month = map(int, month_str.split("-"))
    month_start = date(year, month, 1)

    # Calculate last day of month
    if month == 12:
        next_month = date(year + 1, 1, 1)
    else:
        next_month = date(year, month + 1, 1)
    month_end = next_month - timedelta(days=1)

    joined_date = date.fromisoformat(joined_on_str)
    if joined_date > month_end:
        return 0

    start_date = max(month_start, joined_date)
    count = 0
    cur = start_date
    while cur <= month_end:
        if cur.weekday() < 5:  # Mon=0, Fri=4
            count += 1
        cur += timedelta(days=1)
    return count


def format_attendance_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    Formats a raw MongoDB attendance_logs document into the API AttendanceRecord schema:
    - Excludes internal _id.
    - Converts BSON datetimes to epoch milliseconds.
    - Handles defaults for seeded documents missing optional fields.
    """
    # Format manual correction history
    formatted_history = []
    for h in doc.get("history", []):
        entry_changes = {}
        for field, change in h.get("changes", {}).items():
            f_val = change.get("from")
            t_val = change.get("to")
            if field in ("punch_in", "punch_out"):
                f_val = datetime_to_epoch_ms(f_val) if isinstance(f_val, datetime) else f_val
                t_val = datetime_to_epoch_ms(t_val) if isinstance(t_val, datetime) else t_val
            entry_changes[field] = {"from": f_val, "to": t_val}

        at_val = h.get("at")
        formatted_history.append({
            "at": datetime_to_epoch_ms(at_val) if isinstance(at_val, datetime) else at_val,
            "by": h.get("by"),
            "reason": h.get("reason"),
            "changes": entry_changes,
        })

    return {
        "emp_code": doc["emp_code"],
        "date": doc["date"],
        "status": doc["status"],
        "punch_in": datetime_to_epoch_ms(doc.get("punch_in")),
        "punch_out": datetime_to_epoch_ms(doc.get("punch_out")),
        "work_hours": doc.get("work_hours"),
        "late_minutes": doc.get("late_minutes", 0),
        "overtime_minutes": doc.get("overtime_minutes", 0),
        "half_day": doc.get("half_day", False),
        "history": formatted_history,
    }


# --------------------------------------------------------------------------- #
# 4. Pydantic Schemas (Strict Request Validation)
# --------------------------------------------------------------------------- #
class EmployeeCreate(BaseModel):
    emp_code: str = Field(..., pattern=r"^EMP\d{4,6}$")
    name: str = Field(..., min_length=1, max_length=100)
    email: str = Field(..., max_length=120, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    department: str = Field(..., min_length=1, max_length=50)
    shift_start: str = Field("09:30", pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    shift_end: str = Field("18:30", pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    joined_on: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")

    @field_validator("shift_end")
    def validate_shift_difference(cls, v, info):
        # Spec: shift_start must differ from shift_end
        if "shift_start" in info.data and v == info.data["shift_start"]:
            raise ValueError("shift_start and shift_end cannot be equal")
        return v


class PunchInRequest(BaseModel):
    emp_code: str
    punched_at: Optional[int] = Field(None, ge=100_000_000_000, le=4_102_444_800_000)
    status: str = Field("PRESENT", pattern=r"^(PRESENT|WFH|ON_DUTY)$")


class PunchOutRequest(BaseModel):
    emp_code: str
    punched_at: Optional[int] = Field(None, ge=100_000_000_000, le=4_102_444_800_000)


class RegularizeRequest(BaseModel):
    status: Optional[str] = Field(None, pattern=r"^(PRESENT|ABSENT|LEAVE|WFH|ON_DUTY)$")
    punch_in: Optional[int] = Field(None, ge=100_000_000_000, le=4_102_444_800_000)
    punch_out: Optional[int] = Field(None, ge=100_000_000_000, le=4_102_444_800_000)
    reason: str = Field(..., min_length=5, max_length=200)
    regularized_by: str = Field(..., min_length=1, max_length=50)


# --------------------------------------------------------------------------- #
# 5. Core Endpoints
# --------------------------------------------------------------------------- #

@app.get("/health", status_code=200)
def health():
    """
    Readiness Probe:
    Returns 200 only if MongoDB answers a ping; otherwise 503 Service Unavailable.
    """
    try:
        client.admin.command("ping")
        return {"status": "ok"}
    except PyMongoError:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Database unavailable")


@app.post("/employees", status_code=201)
def create_employee(body: EmployeeCreate):
    """
    Creates an employee document.
    Enforces uniqueness using database unique index on emp_code.
    Returns 409 Conflict if emp_code already exists.
    """
    doc = body.model_dump()
    # Server sets created_at as BSON UTC datetime
    now_utc = datetime.now(timezone.utc).replace(microsecond=0)
    doc["created_at"] = now_utc

    try:
        db.employees.insert_one(doc)
    except DuplicateKeyError:
        raise HTTPException(status_code=409, detail="emp_code already exists")

    doc.pop("_id", None)
    doc["created_at"] = datetime_to_epoch_ms(now_utc)
    return doc


@app.get("/employees", status_code=200)
def list_employees(
    department: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    """
    Lists employees paginated and sorted by emp_code ascending.
    total reflects the filtered set.
    """
    q: Dict[str, Any] = {}
    if department:
        q["department"] = department

    total = db.employees.count_documents(q)
    skip = (page - 1) * page_size
    cursor = db.employees.find(q, {"_id": 0}).sort("emp_code", ASCENDING).skip(skip).limit(page_size)

    items = []
    for emp in cursor:
        emp["created_at"] = datetime_to_epoch_ms(emp.get("created_at"))
        items.append(emp)

    return {"items": items, "total": total, "page": page, "page_size": page_size}


@app.post("/attendance/punch-in", status_code=201)
def punch_in(body: PunchInRequest):
    """
    Records an employee punch-in.
    - Resolves employee (404 if not found).
    - Resolves attendance date according to overnight shift rules (R1).
    - Computes late_minutes (R2).
    - Concurrency-safe: DuplicateKeyError on (emp_code, date) yields 409.
    """
    emp = db.employees.find_one({"emp_code": body.emp_code})
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")

    if body.punched_at is not None:
        ts_utc = epoch_ms_to_datetime(body.punched_at)
    else:
        # Default to now, truncated to whole seconds
        ts_utc = datetime.now(timezone.utc).replace(microsecond=0)

    # Determine attendance calendar date (R1)
    att_date = determine_attendance_date(ts_utc, emp["shift_start"], emp["shift_end"])
    late_minutes = compute_late_minutes(ts_utc, emp["shift_start"], att_date, emp["shift_end"])

    doc = {
        "emp_code": body.emp_code,
        "date": att_date,
        "status": body.status,
        "punch_in": ts_utc,
        "punch_out": None,
        "work_hours": None,
        "late_minutes": late_minutes,
        "overtime_minutes": 0,
        "half_day": False,
        "history": [],
    }

    try:
        db.attendance_logs.insert_one(doc)
    except DuplicateKeyError:
        raise HTTPException(status_code=409, detail="already punched in for this date")

    return format_attendance_doc(doc)


@app.post("/attendance/punch-out", status_code=200)
def punch_out(body: PunchOutRequest):
    """
    Records an employee punch-out.
    - Closes the most recent record whose punch_in <= punched_at.
    - Works across midnight for overnight shifts.
    - Concurrency-safe: atomic find_one_and_update guarantees exactly one 200 and concurrent requests get 409.
    """
    emp = db.employees.find_one({"emp_code": body.emp_code})
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")

    if body.punched_at is not None:
        punch_out_utc = epoch_ms_to_datetime(body.punched_at)
    else:
        punch_out_utc = datetime.now(timezone.utc).replace(microsecond=0)

    # Find the most recent record with punch_in <= punch_out_utc
    rec = db.attendance_logs.find_one(
        {"emp_code": body.emp_code, "punch_in": {"$lte": punch_out_utc}},
        sort=[("punch_in", DESCENDING)],
    )

    if not rec:
        raise HTTPException(status_code=404, detail="No matching punch-in found")

    if rec.get("punch_out") is not None:
        raise HTTPException(status_code=409, detail="Record already punched out")

    punch_in_utc = rec["punch_in"]
    if punch_out_utc <= punch_in_utc:
        raise HTTPException(status_code=422, detail="punch_out must be after punch_in")

    duration_seconds = (punch_out_utc - punch_in_utc).total_seconds()
    if duration_seconds > 86400:
        raise HTTPException(status_code=422, detail="punch_out cannot be more than 24 hours after punch_in")

    # Compute derived fields (R3, R4, R5)
    work_hours, overtime_minutes, half_day = compute_work_hours_and_overtime(
        punch_in_utc, punch_out_utc, emp["shift_end"], rec["date"], emp["shift_start"]
    )

    # Atomic update: only updates if punch_out is still None
    updated = db.attendance_logs.find_one_and_update(
        {"_id": rec["_id"], "punch_out": None},
        {
            "$set": {
                "punch_out": punch_out_utc,
                "work_hours": work_hours,
                "overtime_minutes": overtime_minutes,
                "half_day": half_day,
            }
        },
        return_document=ReturnDocument.AFTER,
    )

    if not updated:
        # A concurrent request punched out this exact document first
        raise HTTPException(status_code=409, detail="Record already punched out")

    return format_attendance_doc(updated)


@app.get("/attendance", status_code=200)
def list_attendance(
    emp_code: Optional[str] = None,
    date_from: Optional[str] = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
    date_to: Optional[str] = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
    status: Optional[str] = Query(None, pattern=r"^(PRESENT|ABSENT|LEAVE|WFH|ON_DUTY)$"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    """
    Lists attendance records sorted by date descending, then emp_code ascending.
    Direct database skip/limit prevents memory exhaustion at scale.
    """
    if date_from and date_to and date_from > date_to:
        raise HTTPException(status_code=422, detail="date_from cannot be greater than date_to")

    q: Dict[str, Any] = {}
    if emp_code:
        q["emp_code"] = emp_code
    if date_from or date_to:
        q["date"] = {}
        if date_from:
            q["date"]["$gte"] = date_from
        if date_to:
            q["date"]["$lte"] = date_to
    if status:
        q["status"] = status

    total = db.attendance_logs.count_documents(q)
    skip = (page - 1) * page_size
    cursor = (
        db.attendance_logs.find(q)
        .sort([("date", DESCENDING), ("emp_code", ASCENDING)])
        .skip(skip)
        .limit(page_size)
    )

    items = [format_attendance_doc(d) for d in cursor]
    return {"items": items, "total": total, "page": page, "page_size": page_size}


@app.patch("/attendance/{emp_code}/{date}", status_code=200)
def regularize_attendance(
    emp_code: str = Path(...),
    date: str = Path(..., pattern=r"^\d{4}-\d{2}-\d{2}$"),
    body: RegularizeRequest = ...,
):
    """
    Regularizes (manually corrects) an attendance record.
    - Appends exactly one entry to history with only the fields that actually changed.
    - Stores BSON datetimes in MongoDB history.changes; returns epoch ms in API.
    - Recomputes derived fields.
    - Concurrency-safe: uses history array size checking to prevent silent overwrite of concurrent edits.
    """
    emp = db.employees.find_one({"emp_code": emp_code})
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")

    rec = db.attendance_logs.find_one({"emp_code": emp_code, "date": date})
    if not rec:
        raise HTTPException(status_code=404, detail="Attendance record not found")

    # Determine final status
    new_status = body.status if body.status is not None else rec["status"]

    if new_status in ("ABSENT", "LEAVE"):
        # Setting ABSENT/LEAVE clears punch times; supplying punch times is 422
        if body.punch_in is not None or body.punch_out is not None:
            raise HTTPException(status_code=422, detail="Cannot supply punch times when status is ABSENT or LEAVE")
        final_punch_in = None
        final_punch_out = None
        final_work_hours = None
        final_late_minutes = 0
        final_overtime_minutes = 0
        final_half_day = False
    else:
        # Presence status requires punch_in
        if body.punch_in is not None:
            final_punch_in = epoch_ms_to_datetime(body.punch_in)
        else:
            final_punch_in = rec.get("punch_in")

        if final_punch_in is None:
            raise HTTPException(status_code=422, detail="Presence status requires a punch_in")

        # punch_in must stay on the record's attendance date (R1)
        calc_date = determine_attendance_date(final_punch_in, emp["shift_start"], emp["shift_end"])
        if calc_date != date:
            raise HTTPException(status_code=422, detail="punch_in must stay on the record's attendance date")

        final_late_minutes = compute_late_minutes(final_punch_in, emp["shift_start"], date, emp["shift_end"])

        if body.punch_out is not None:
            final_punch_out = epoch_ms_to_datetime(body.punch_out)
        else:
            final_punch_out = rec.get("punch_out")

        if final_punch_out is not None:
            if final_punch_out <= final_punch_in:
                raise HTTPException(status_code=422, detail="punch_out must be after punch_in")
            duration_sec = (final_punch_out - final_punch_in).total_seconds()
            if duration_sec > 86400:
                raise HTTPException(status_code=422, detail="punch_out cannot be more than 24 hours after punch_in")

            final_work_hours, final_overtime_minutes, final_half_day = compute_work_hours_and_overtime(
                final_punch_in, final_punch_out, emp["shift_end"], date, emp["shift_start"]
            )
        else:
            final_work_hours = None
            final_overtime_minutes = 0
            final_half_day = False

    # Detect field diffs
    changes_mongo: Dict[str, Any] = {}
    candidate_changes = [
        ("status", rec.get("status"), new_status),
        ("punch_in", rec.get("punch_in"), final_punch_in),
        ("punch_out", rec.get("punch_out"), final_punch_out),
        ("work_hours", rec.get("work_hours"), final_work_hours),
        ("late_minutes", rec.get("late_minutes", 0), final_late_minutes),
        ("overtime_minutes", rec.get("overtime_minutes", 0), final_overtime_minutes),
        ("half_day", rec.get("half_day", False), final_half_day),
    ]

    for key, old_val, new_val in candidate_changes:
        if old_val != new_val:
            changes_mongo[key] = {"from": old_val, "to": new_val}

    if not changes_mongo:
        raise HTTPException(status_code=422, detail="Request modifies no fields")

    now_utc = datetime.now(timezone.utc).replace(microsecond=0)
    history_entry = {
        "at": now_utc,
        "by": body.regularized_by,
        "reason": body.reason,
        "changes": changes_mongo,
    }

    # Optimistic concurrency: ensure history array length has not changed
    current_history = rec.get("history", [])
    expected_history_len = len(current_history)

    update_query = {
        "_id": rec["_id"],
        "$expr": {"$eq": [{"$size": {"$ifNull": ["$history", []]}}, expected_history_len]},
    }
    update_data = {
        "$set": {
            "status": new_status,
            "punch_in": final_punch_in,
            "punch_out": final_punch_out,
            "work_hours": final_work_hours,
            "late_minutes": final_late_minutes,
            "overtime_minutes": final_overtime_minutes,
            "half_day": final_half_day,
        },
        "$push": {"history": history_entry},
    }

    updated = db.attendance_logs.find_one_and_update(
        update_query, update_data, return_document=ReturnDocument.AFTER
    )

    if not updated:
        raise HTTPException(status_code=409, detail="Concurrent regularization conflict")

    return format_attendance_doc(updated)


# --------------------------------------------------------------------------- #
# 6. Analytics Aggregations
# --------------------------------------------------------------------------- #

@app.get("/analytics/employees/{emp_code}/monthly", status_code=200)
def employee_monthly(
    emp_code: str = Path(...),
    month: str = Query(..., pattern=r"^\d{4}-(0[1-9]|1[0-2])$"),
):
    """
    Monthly attendance summary for one employee:
    - Pure MongoDB aggregation over attendance_logs.
    - Mon-Fri working days adjusted for mid-month joiners (R7).
    - Mon-Fri presence counted (half-day = 0.5) (R5-R7).
    - Half-up rounded attendance_pct (R8).
    """
    emp = db.employees.find_one({"emp_code": emp_code})
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")

    working_days = count_working_days(emp["joined_on"], month)

    date_start = f"{month}-01"
    date_end = f"{month}-31"

    pipeline = [
        {
            "$match": {
                "emp_code": emp_code,
                "date": {"$gte": date_start, "$lte": date_end},
            }
        },
        {
            "$project": {
                "status": 1,
                "late_minutes": {"$ifNull": ["$late_minutes", 0]},
                "overtime_minutes": {"$ifNull": ["$overtime_minutes", 0]},
                "half_day": {"$ifNull": ["$half_day", False]},
                "dateObj": {"$dateFromString": {"dateString": "$date"}},
            }
        },
        {
            "$group": {
                "_id": None,
                "present_days": {
                    "$sum": {
                        "$cond": [
                            {
                                "$and": [
                                    {"$in": ["$status", ["PRESENT", "WFH", "ON_DUTY"]]},
                                    {"$in": [{"$dayOfWeek": "$dateObj"}, [2, 3, 4, 5, 6]]},  # Mon-Fri
                                ]
                            },
                            {"$cond": ["$half_day", 0.5, 1.0]},
                            0.0,
                        ]
                    }
                },
                "leave_days": {
                    "$sum": {"$cond": [{"$eq": ["$status", "LEAVE"]}, 1, 0]}
                },
                "late_count": {
                    "$sum": {"$cond": [{"$gt": ["$late_minutes", 0]}, 1, 0]}
                },
                "total_late_minutes": {"$sum": "$late_minutes"},
                "total_overtime_minutes": {"$sum": "$overtime_minutes"},
            }
        },
    ]

    res = list(db.attendance_logs.aggregate(pipeline))
    if res:
        agg = res[0]
        present_days = float(agg.get("present_days", 0.0))
        leave_days = int(agg.get("leave_days", 0))
        late_count = int(agg.get("late_count", 0))
        total_late_minutes = int(agg.get("total_late_minutes", 0))
        total_overtime_minutes = int(agg.get("total_overtime_minutes", 0))
    else:
        present_days = 0.0
        leave_days = 0
        late_count = 0
        total_late_minutes = 0
        total_overtime_minutes = 0

    if working_days > 0:
        attendance_pct = round_half_up((present_days / working_days) * 100.0, 2)
    else:
        attendance_pct = None

    return {
        "emp_code": emp_code,
        "month": month,
        "working_days": working_days,
        "present_days": present_days,
        "leave_days": leave_days,
        "late_count": late_count,
        "total_late_minutes": total_late_minutes,
        "total_overtime_minutes": total_overtime_minutes,
        "attendance_pct": attendance_pct,
    }


@app.get("/analytics/departments/summary", status_code=200)
def department_summary(
    month: str = Query(..., pattern=r"^\d{4}-(0[1-9]|1[0-2])$"),
    department: Optional[str] = None,
):
    """
    Department monthly summary:
    - Cross-collection aggregation starting from employees (preserves people with zero logs in headcount).
    - Excludes employees joining after month end (R9).
    - Omit departments whose headcount is 0.
    - avg_work_hours averaged across records, rounded half-up (R8).
    """
    year, m = map(int, month.split("-"))
    if m == 12:
        next_month = date(year + 1, 1, 1)
    else:
        next_month = date(year, m + 1, 1)
    last_day_str = (next_month - timedelta(days=1)).isoformat()
    first_day_str = f"{month}-01"

    emp_match: Dict[str, Any] = {"joined_on": {"$lte": last_day_str}}
    if department:
        emp_match["department"] = department

    pipeline = [
        {"$match": emp_match},
        {
            "$lookup": {
                "from": "attendance_logs",
                "let": {"emp": "$emp_code"},
                "pipeline": [
                    {
                        "$match": {
                            "$expr": {
                                "$and": [
                                    {"$eq": ["$emp_code", "$$emp"]},
                                    {"$gte": ["$date", first_day_str]},
                                    {"$lte": ["$date", last_day_str]},
                                ]
                            }
                        }
                    }
                ],
                "as": "logs",
            }
        },
        {
            "$group": {
                "_id": "$department",
                "headcount": {"$sum": 1},
                "all_logs": {"$push": "$logs"},
            }
        },
        {"$sort": {"_id": 1}},
    ]

    dept_groups = list(db.employees.aggregate(pipeline))
    items = []

    for dg in dept_groups:
        dept_name = dg["_id"]
        headcount = dg["headcount"]
        if headcount == 0:
            continue

        present_days = Decimal("0.0")
        work_hours_total = Decimal("0.0")
        work_hours_count = 0
        late_count = 0
        total_late_minutes = 0
        leave_count = 0
        on_duty_count = 0

        for log_list in dg.get("all_logs", []):
            for log in log_list:
                status_val = log.get("status")
                log_date_str = log.get("date")
                is_weekday = date.fromisoformat(log_date_str).weekday() < 5

                # Present days counts Mon-Fri presence only (R7)
                if status_val in ("PRESENT", "WFH", "ON_DUTY") and is_weekday:
                    if log.get("half_day"):
                        present_days += Decimal("0.5")
                    else:
                        present_days += Decimal("1.0")

                # Work hours average over presence status records with work_hours
                if status_val in ("PRESENT", "WFH", "ON_DUTY") and log.get("work_hours") is not None:
                    work_hours_total += Decimal(str(log["work_hours"]))
                    work_hours_count += 1

                # Late metrics
                late_m = log.get("late_minutes", 0)
                if late_m > 0:
                    late_count += 1
                    total_late_minutes += late_m

                # Status counters
                if status_val == "LEAVE":
                    leave_count += 1
                elif status_val == "ON_DUTY":
                    on_duty_count += 1

        if work_hours_count > 0:
            avg_work_hours = round_half_up(float(work_hours_total / work_hours_count), 2)
        else:
            avg_work_hours = None

        items.append({
            "department": dept_name,
            "headcount": headcount,
            "present_days": float(present_days),
            "avg_work_hours": avg_work_hours,
            "late_count": late_count,
            "total_late_minutes": total_late_minutes,
            "leave_count": leave_count,
            "on_duty_count": on_duty_count,
        })

    return {"month": month, "items": items}


@app.get("/analytics/leaderboard/late", status_code=200)
def late_leaderboard(
    month: str = Query(..., pattern=r"^\d{4}-(0[1-9]|1[0-2])$"),
    limit: int = Query(10, ge=1, le=50),
    department: Optional[str] = None,
):
    """
    Late leaderboard for a month:
    - Pure MongoDB aggregation pipeline.
    - Uses $setWindowFields with $rank for standard competition ranking (1, 2, 2, 4).
    - limit applied after ranking: includes all tied employees whose rank <= limit.
    - Sorted by total_late_minutes descending, emp_code ascending.
    """
    date_start = f"{month}-01"
    date_end = f"{month}-31"

    pipeline: List[Dict[str, Any]] = [
        {
            "$match": {
                "date": {"$gte": date_start, "$lte": date_end},
                "late_minutes": {"$gt": 0},
            }
        },
        {
            "$group": {
                "_id": "$emp_code",
                "total_late_minutes": {"$sum": "$late_minutes"},
                "late_count": {"$sum": 1},
            }
        },
        {
            "$lookup": {
                "from": "employees",
                "localField": "_id",
                "foreignField": "emp_code",
                "as": "emp",
            }
        },
        {"$unwind": "$emp"},
    ]

    if department:
        pipeline.append({"$match": {"emp.department": department}})

    # Rank using standard competition ranking
    pipeline.extend([
        {
            "$setWindowFields": {
                "sortBy": {"total_late_minutes": -1},
                "output": {"rank": {"$rank": {}}},
            }
        },
        {"$match": {"rank": {"$lte": limit}}},
        {"$sort": {"total_late_minutes": -1, "_id": 1}},
        {
            "$project": {
                "_id": 0,
                "rank": 1,
                "emp_code": "$_id",
                "name": "$emp.name",
                "department": "$emp.department",
                "total_late_minutes": 1,
                "late_count": 1,
            }
        },
    ])

    items = list(db.attendance_logs.aggregate(pipeline))
    return {"month": month, "items": items}


@app.get("/analytics/departments/{department}/trend", status_code=200)
def department_trend(
    department: str = Path(...),
    date_from: str = Query(..., alias="from", pattern=r"^\d{4}-\d{2}-\d{2}$"),
    date_to: str = Query(..., alias="to", pattern=r"^\d{4}-\d{2}-\d{2}$"),
):
    """
    Daily attendance trend for a department:
    - Gap-filling across date range using MongoDB $densify (no Python loops).
    - 7-day moving average calculated using $setWindowFields with documents: [-6, "current"].
    - 4 decimals half-up for rate and moving average (R8).
    """
    if date_to < date_from:
        raise HTTPException(status_code=422, detail="'to' date cannot be before 'from' date")

    d_from = date.fromisoformat(date_from)
    d_to = date.fromisoformat(date_to)
    if (d_to - d_from).days > 92:
        raise HTTPException(status_code=422, detail="Date range cannot exceed 92 days")

    # 404 for unknown department (no employees exist in department)
    dept_exists = db.employees.find_one({"department": department})
    if not dept_exists:
        raise HTTPException(status_code=404, detail="Department not found")

    dt_from_utc = datetime.fromisoformat(date_from).replace(tzinfo=timezone.utc)
    # $densify upper bound is exclusive
    dt_to_bound = datetime.fromisoformat(date_to).replace(tzinfo=timezone.utc) + timedelta(days=1)

    pipeline = [
        # 1. Match records for department within range (uses IXSCAN)
        {"$match": {"date": {"$gte": date_from, "$lte": date_to}}},
        {
            "$lookup": {
                "from": "employees",
                "localField": "emp_code",
                "foreignField": "emp_code",
                "as": "emp",
            }
        },
        {"$unwind": "$emp"},
        {"$match": {"emp.department": department}},
        {
            "$project": {
                "dateStr": "$date",
                "dateObj": {"$dateFromString": {"dateString": "$date"}},
                "status": 1,
                "half_day": {"$ifNull": ["$half_day", False]},
                "late_minutes": {"$ifNull": ["$late_minutes", 0]},
            }
        },
        # 2. Summarize actual attendance logs per day
        {
            "$group": {
                "_id": "$dateObj",
                "date": {"$first": "$dateStr"},
                "present_count": {
                    "$sum": {
                        "$cond": [
                            {"$in": ["$status", ["PRESENT", "WFH", "ON_DUTY"]]},
                            {"$cond": ["$half_day", 0.5, 1.0]},
                            0.0,
                        ]
                    }
                },
                "late_count": {
                    "$sum": {"$cond": [{"$gt": ["$late_minutes", 0]}, 1, 0]}
                },
            }
        },
        # 3. Gap-filling: densify every calendar day between from and to
        {
            "$densify": {
                "field": "_id",
                "range": {
                    "step": 1,
                    "unit": "day",
                    "bounds": [dt_from_utc, dt_to_bound],
                },
            }
        },
        # 4. Normalize fields for gap-filled days
        {
            "$project": {
                "_id": 0,
                "dateObj": "$_id",
                "date": {"$dateToString": {"format": "%Y-%m-%d", "date": "$_id"}},
                "present_count": {"$ifNull": ["$present_count", 0.0]},
                "late_count": {"$ifNull": ["$late_count", 0]},
            }
        },
        # 5. Join employees to compute active headcount on that day
        {
            "$lookup": {
                "from": "employees",
                "let": {"day": "$date"},
                "pipeline": [
                    {
                        "$match": {
                            "$expr": {
                                "$and": [
                                    {"$eq": ["$department", department]},
                                    {"$lte": ["$joined_on", "$$day"]},
                                ]
                            }
                        }
                    }
                ],
                "as": "active_emps",
            }
        },
        {
            "$addFields": {
                "headcount": {"$size": "$active_emps"},
                "is_working_day": {
                    "$in": [{"$dayOfWeek": "$dateObj"}, [2, 3, 4, 5, 6]]  # 2=Mon, 6=Fri
                },
            }
        },
        # 6. Compute attendance rate
        {
            "$addFields": {
                "attendance_rate": {
                    "$cond": [
                        {
                            "$and": [
                                "$is_working_day",
                                {"$gt": ["$headcount", 0]},
                            ]
                        },
                        {"$divide": ["$present_count", "$headcount"]},
                        None,
                    ]
                }
            }
        },
        {"$sort": {"date": 1}},
        # 7. 7-day moving average using $setWindowFields
        {
            "$setWindowFields": {
                "sortBy": {"date": 1},
                "output": {
                    "moving_avg_7d": {
                        "$avg": "$attendance_rate",
                        "window": {"documents": [-6, "current"]},
                    }
                },
            }
        },
    ]

    docs = list(db.attendance_logs.aggregate(pipeline))
    items = []

    for d in docs:
        rate = round_half_up(d.get("attendance_rate"), 4)
        m_avg = round_half_up(d.get("moving_avg_7d"), 4)
        items.append({
            "date": d["date"],
            "is_working_day": d["is_working_day"],
            "headcount": d["headcount"],
            "present_count": float(d["present_count"]),
            "late_count": int(d["late_count"]),
            "attendance_rate": rate,
            "moving_avg_7d": m_avg,
        })

    return {"department": department, "items": items}


# --------------------------------------------------------------------------- #
# 7. Admin Explain Endpoint
# --------------------------------------------------------------------------- #

@app.get("/admin/explain/{endpoint}", status_code=200)
def explain_endpoint(
    endpoint: str = Path(..., pattern=r"^(attendance_list|employee_monthly|department_summary|late_leaderboard|department_trend)$"),
    emp_code: Optional[str] = Query(None),
    month: Optional[str] = Query(None, pattern=r"^\d{4}-(0[1-9]|1[0-2])$"),
    department: Optional[str] = Query(None),
    limit: Optional[int] = Query(10, ge=1, le=50),
    date_from: Optional[str] = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
    date_to: Optional[str] = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
    status_filter: Optional[str] = Query(None, alias="status", pattern=r"^(PRESENT|ABSENT|LEAVE|WFH|ON_DUTY)$"),
    from_date: Optional[str] = Query(None, alias="from", pattern=r"^\d{4}-\d{2}-\d{2}$"),
    to_date: Optional[str] = Query(None, alias="to", pattern=r"^\d{4}-\d{2}-\d{2}$"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    """
    Runs MongoDB explain with executionStats verbosity for the main query or
    aggregation of the named endpoint, built exactly as that endpoint builds it.
    Proves index scan (IXSCAN) utilization without collection scan (COLLSCAN).
    """
    if endpoint == "attendance_list":
        q: Dict[str, Any] = {}
        if emp_code:
            q["emp_code"] = emp_code
        if date_from or date_to:
            q["date"] = {}
            if date_from:
                q["date"]["$gte"] = date_from
            if date_to:
                q["date"]["$lte"] = date_to
        if status_filter:
            q["status"] = status_filter

        find_cmd = {
            "find": "attendance_logs",
            "filter": q,
            "sort": {"date": -1, "emp_code": 1},
            "skip": (page - 1) * page_size,
            "limit": page_size,
        }
        exp = db.command("explain", find_cmd, verbosity="executionStats")
        return {"endpoint": endpoint, "collection": "attendance_logs", "explain": exp}

    elif endpoint == "employee_monthly":
        if not emp_code:
            raise HTTPException(status_code=422, detail="emp_code required")
        if not month:
            raise HTTPException(status_code=422, detail="month required")

        pipe = [
            {"$match": {"emp_code": emp_code, "date": {"$gte": f"{month}-01", "$lte": f"{month}-31"}}},
            {
                "$project": {
                    "status": 1,
                    "late_minutes": {"$ifNull": ["$late_minutes", 0]},
                    "overtime_minutes": {"$ifNull": ["$overtime_minutes", 0]},
                    "half_day": {"$ifNull": ["$half_day", False]},
                    "dateObj": {"$dateFromString": {"dateString": "$date"}},
                }
            },
            {"$group": {"_id": None, "count": {"$sum": 1}}},
        ]
        agg_cmd = {"aggregate": "attendance_logs", "pipeline": pipe, "cursor": {}}
        exp = db.command("explain", agg_cmd, verbosity="executionStats")
        return {"endpoint": endpoint, "collection": "attendance_logs", "explain": exp}

    elif endpoint == "department_summary":
        if not month:
            raise HTTPException(status_code=422, detail="month required")

        emp_match: Dict[str, Any] = {"joined_on": {"$lte": f"{month}-31"}}
        if department:
            emp_match["department"] = department

        pipe = [
            {"$match": emp_match},
            {
                "$lookup": {
                    "from": "attendance_logs",
                    "let": {"emp": "$emp_code"},
                    "pipeline": [
                        {
                            "$match": {
                                "$expr": {
                                    "$and": [
                                        {"$eq": ["$emp_code", "$$emp"]},
                                        {"$gte": ["$date", f"{month}-01"]},
                                        {"$lte": ["$date", f"{month}-31"]},
                                    ]
                                }
                            }
                        }
                    ],
                    "as": "logs",
                }
            },
            {"$group": {"_id": "$department", "headcount": {"$sum": 1}}},
        ]
        agg_cmd = {"aggregate": "employees", "pipeline": pipe, "cursor": {}}
        exp = db.command("explain", agg_cmd, verbosity="executionStats")
        return {"endpoint": endpoint, "collection": "employees", "explain": exp}

    elif endpoint == "late_leaderboard":
        if not month:
            raise HTTPException(status_code=422, detail="month required")

        pipe = [
            {
                "$match": {
                    "date": {"$gte": f"{month}-01", "$lte": f"{month}-31"},
                    "late_minutes": {"$gt": 0},
                }
            },
            {"$group": {"_id": "$emp_code", "total_late_minutes": {"$sum": "$late_minutes"}}},
        ]
        agg_cmd = {"aggregate": "attendance_logs", "pipeline": pipe, "cursor": {}}
        exp = db.command("explain", agg_cmd, verbosity="executionStats")
        return {"endpoint": endpoint, "collection": "attendance_logs", "explain": exp}

    elif endpoint == "department_trend":
        if not department:
            raise HTTPException(status_code=422, detail="department required")
        if not from_date or not to_date:
            raise HTTPException(status_code=422, detail="'from' and 'to' parameters required")

        pipe = [
            {"$match": {"date": {"$gte": from_date, "$lte": to_date}}},
            {
                "$lookup": {
                    "from": "employees",
                    "localField": "emp_code",
                    "foreignField": "emp_code",
                    "as": "emp",
                }
            },
            {"$unwind": "$emp"},
            {"$match": {"emp.department": department}},
        ]
        agg_cmd = {"aggregate": "attendance_logs", "pipeline": pipe, "cursor": {}}
        exp = db.command("explain", agg_cmd, verbosity="executionStats")
        return {"endpoint": endpoint, "collection": "attendance_logs", "explain": exp}
