# DECISIONS.md 

1. **Indexes.**
   On `employees`, we created a unique index on `emp_code` to enforce business identity, and `(department, joined_on)` for headcount filtering. On `attendance_logs`, we created a unique compound index on `(emp_code, date)` to guarantee one record per employee per day and eliminate punch-in races. We added `(date, emp_code)` for fast listing, `(date, late_minutes)` for leaderboard filtering, and `(emp_code, punch_in)` for punch-out lookups. We rejected indexing `department` on attendance logs because departments live on employees and can change over time.

2. **Punch-in race.**
   Both concurrent requests resolve employee shift details and compute the attendance date. Both attempt `insert_one()`. MongoDB's unique index on `(emp_code, date)` allows exactly one document to insert successfully (returning 201). The second request raises a `DuplicateKeyError`, which our code intercepts to immediately return HTTP 409 Conflict.

3. **Ties.**
   Our pipeline applies MongoDB's `$rank` window operator to assign standard competition ranks (1, 2, 2, 4). Because the contract requires filtering by `rank <= limit` (rather than slicing an array), both tied employees at the cutoff rank are returned, breaking row ties by `emp_code`. The response will therefore contain more than `limit` rows.

4. **Headcount.**
   The summary aggregation starts from `employees` filtered by `joined_on <= end_of_month` and uses `$lookup` to join matching monthly attendance logs. Department headcount is counted directly from the matched employee documents, so employees with zero attendance records are fully preserved in headcount.

5. **One thing you would change for 100x data.**
   Precompute and persist daily department rollups via MongoDB change streams or background jobs, and shard `attendance_logs` on a hashed or compound key (`date`, `emp_code`), avoiding expensive on-the-fly joins and `$densify` stages over tens of millions of records.
