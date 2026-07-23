---
name: surql
description: >
  Authoritative SurrealQL v3 (3.1+) syntax and best-practice manual. Dense, code-first
  reference for schema definition (tables, fields, indexes, analyzers), relationships
  (record links and graph edges), computed and event-driven derived data, record-access
  authentication, full-text search, sequences, live queries, transactions, the HTTP /sql
  API, and capabilities. Load whenever writing, reviewing, or translating SurrealQL so
  syntax is copied, not invented. Pairs with the SurrealQL Manifest, which holds the rules;
  this file holds the syntax.
---

# [[surql]] [[agent#skills|skills]]

> Targets **SurrealDB v3, current line (3.1+)**. Auth and tests run over HTTP `/sql`. Experimental features (`surrealism`/WASM, `files`/buckets) are off. Every snippet below is copy-ready v3 syntax. This is the *how*; the Manifest is the *rules*.

---

## 1. Data types & literals

| Type | Literal / example | Notes |
|---|---|---|
| `string` | `'hello'`, `"hi"` | |
| `int` / `float` | `42`, `3.14` | |
| `decimal` | `19.99dec`, `18500000dec` | exact; use for money & precise quantities |
| `bool` | `true`, `false` | |
| `datetime` | `d'2026-01-15T10:00:00Z'` | always UTC ISO-8601 |
| `duration` | `1h`, `30m`, `2w`, `500ms`, `0ns` | |
| `uuid` | `u'01890 a1b...'`, `rand::uuid()` | |
| `record<T>` | `user:jane`, `user:⟨ulid⟩` | a typed link to another record |
| `object` | `{ a: 1, b: 'x' }` | |
| `array<T>` | `[1, 2, 3]`, `['a', 'b']` | also `set<T>` for de-duplicated |
| `geometry<point>` | `(-3.70, 40.41)` | also `polygon`, `line`, `multipoint`, … |
| `bytes` | `<bytes>"..."` | |
| `option<T>` | — | nullable wrapper; everything else is required |
| literal union | `'a' \| 'b'`, `{k:'x'} \| {k:'y'}` | discriminated, engine-validated |
| `none` / `null` | `NONE`, `NULL` | `NONE` = absent, `NULL` = explicit null |

Useful operators: `??` (null-coalesce — first non-NULL/NONE), `OR` (value-coalesce — first truthy, e.g. `$x OR 0`), `?:` (empty-coalesce), `+=` / `-=` (in-place), `CONTAINS`/`INSIDE`/`OUTSIDE` (collections & geometry), `@@` (full-text match), `IS`/`IS NOT`, `INSIDE` for point-in-polygon.

---

## 2. Record IDs

```surql
user:jane                       -- explicit, meaningful id
CREATE user;                    -- auto id (random)
CREATE user SET …;              -- auto id
CREATE user:`my id`;            -- id needing quoting
CREATE order:⟨ulid()⟩;          -- generated sortable id  (also rand::ulid())
type::thing('user', $id);       -- build a record id from parts
record::id(user:jane);          -- -> 'jane'
```

Composite / array ids: `user:[2026, 'jane']`. IDs are the join key — there are no foreign keys, only links and edges.

**Record ranges** — query a contiguous slice of IDs directly (a partition scan; no index needed, since the ID *is* the pointer to the data):

```surql
SELECT * FROM user:1000..=2000;          -- inclusive both ends  ( ..= )
SELECT * FROM user:1000..2000;           -- exclusive upper end  ( ..  )
SELECT * FROM user:1000..;               -- open upper bound
SELECT * FROM user:..=2000;              -- open lower bound
-- composite/partition key: pin the prefix, range over the trailing part
SELECT * FROM weather:['London', $from]..=['London', $to];
```

> A range over a composite ID beats a `WHERE location = … AND at > … AND at < …` (+ compound index): it touches only the matching partition and is often >100× faster on large append-only tables. See §25.

---

## 3. DEFINE TABLE

```surql
DEFINE TABLE [OVERWRITE | IF NOT EXISTS] name
  [ DROP ]
  [ SCHEMAFULL | SCHEMALESS ]
  [ TYPE ANY | NORMAL | RELATION [IN in_tables] [OUT out_tables] [ENFORCED] ]
  [ AS SELECT … ]                         -- pre-computed (auto-updating) view
  [ CHANGEFEED duration [INCLUDE ORIGINAL] ]
  [ PERMISSIONS NONE | FULL | FOR select|create|update|delete WHERE … ]
  [ COMMENT '…' ];
```

Standard entity table (always schemafull, explicit permissions):

```surql
DEFINE TABLE user SCHEMAFULL
  PERMISSIONS
    FOR select, update, delete WHERE id = $auth.id
    FOR create WHERE $auth.id != NONE;
```

Relation (edge) table — `ENFORCED` requires the in/out records to exist:

```surql
DEFINE TABLE authored SCHEMAFULL TYPE RELATION IN user OUT post ENFORCED
  PERMISSIONS FOR select FULL;
```

Pre-computed view (single source only; auto-maintained on writes):

```surql
DEFINE TABLE posts_by_author AS
  SELECT count() AS total, author FROM post GROUP BY author;
```

---

## 4. DEFINE FIELD

Full clause order:

```surql
DEFINE FIELD [OVERWRITE | IF NOT EXISTS] name ON [TABLE] table
  [ [FLEXIBLE] TYPE type | COMPUTED expression ]
  [ DEFAULT [ALWAYS] expression ]
  [ VALUE expression ]
  [ ASSERT expression ]
  [ READONLY ]
  [ REFERENCE [ ON DELETE REJECT | CASCADE | UNSET | IGNORE | THEN expr ] ]
  [ PERMISSIONS … ]
  [ COMMENT '…' ];
```

Common patterns:

```surql
-- scalars & constraints
DEFINE FIELD email   ON user TYPE string ASSERT string::is_email($value);
DEFINE FIELD age     ON user TYPE int ASSERT $value >= 0;
DEFINE FIELD bio     ON user TYPE option<string>;            -- nullable
DEFINE FIELD role    ON user TYPE string
  ASSERT $value IN ['admin', 'staff', 'customer'] DEFAULT 'customer';  -- enum

-- timestamps (auto-maintained — created stays put, updated refreshes every write)
DEFINE FIELD created_at ON user TYPE datetime DEFAULT time::now() READONLY;
DEFINE FIELD updated_at ON user TYPE datetime VALUE time::now();

-- nested object + typed open map
DEFINE FIELD address       ON user TYPE object;
DEFINE FIELD address.city  ON user TYPE string;
DEFINE FIELD address.zip   ON user TYPE string;
DEFINE FIELD meta          ON user TYPE object;
DEFINE FIELD meta.*        ON user TYPE string;              -- open keys, typed values

-- arrays + element constraint
DEFINE FIELD tags    ON user TYPE array<string>;
DEFINE FIELD tags.*  ON user TYPE string;

-- links with explicit referential action (never rely on default IGNORE)
DEFINE FIELD author  ON post TYPE record<user> REFERENCE ON DELETE CASCADE;
DEFINE FIELD org     ON user TYPE record<org>  REFERENCE ON DELETE REJECT;

-- computed (intra-record only; evaluated at read, never stored)
DEFINE FIELD full_name ON user    COMPUTED string::concat(first, ' ', last);
DEFINE FIELD total     ON line     COMPUTED price * quantity;

-- typed flexibility for heterogeneous payloads (literal union — still validated)
DEFINE FIELD event_data ON activity TYPE
    { kind: 'click', x: int, y: int }
  | { kind: 'view',  url: string };

-- constrain the elements of an array with a closure
DEFINE FIELD langs ON business TYPE array<string>
  ASSERT array::all($value, |$l| $l IN ['es', 'en', 'fr', 'ja', 'de', 'pt', 'zh']);
```

> Fields are computed in **alphabetical order**. A computed field depending on another must sort after it. `VALUE` runs on every write; `DEFAULT` only on create; `DEFAULT ALWAYS` re-applies the default on update when the field is unset.

---

## 5. DEFINE INDEX

```surql
DEFINE INDEX [OVERWRITE | IF NOT EXISTS] name ON [TABLE] table FIELDS f1, f2
  [ UNIQUE
  | FULLTEXT ANALYZER analyzer BM25 [HIGHLIGHTS]      -- full-text (one field per index)
  | HNSW DIMENSION n [DIST COSINE|EUCLIDEAN|…] [TYPE F32]   -- vector ANN
  ]
  [ CONCURRENTLY ];
```

```surql
DEFINE INDEX user_email   ON user  FIELDS email UNIQUE;
DEFINE INDEX post_author  ON post  FIELDS author;                 -- speed up WHERE/ORDER
DEFINE INDEX edge_unique  ON authored FIELDS in, out UNIQUE;      -- one edge per pair
DEFINE INDEX post_title_fts ON post FIELDS title FULLTEXT ANALYZER text BM25 HIGHLIGHTS;
```

> Vectors: `HNSW` (or `DISKANN`, new in 3.1) — treat as a noted, revisit-later capability, not a default. `ALTER INDEX` / `REBUILD INDEX` exist for safe maintenance. Computed fields cannot be indexed.

---

## 6. DEFINE ANALYZER (full-text)

```surql
DEFINE ANALYZER text
  TOKENIZERS class
  FILTERS lowercase, ascii, snowball(english);

DEFINE ANALYZER autocomplete
  TOKENIZERS class
  FILTERS lowercase, edgengram(2, 10);
```

Tokenizers: `blank`, `class`, `camel`, `punct`. Filters: `lowercase`, `uppercase`, `ascii`, `snowball(lang)`, `ngram(min,max)`, `edgengram(min,max)`, `mapper(path)`.

---

## 7. Relationships

### 7a. Record links — for 1:N / N:1 with no link data

```surql
DEFINE FIELD author ON post TYPE record<user> REFERENCE ON DELETE CASCADE;

SELECT *, author.name FROM post;          -- dot access dereferences
SELECT * FROM post FETCH author;          -- inline the full linked record
SELECT * FROM post FETCH author, comments;
```

### 7b. Array of links — small bounded set, no per-link data

```surql
DEFINE FIELD tags ON post TYPE array<record<tag>>;
SELECT * FROM post FETCH tags;
```

### 7c. Graph edges — M:N, OR carries data, OR traversed

```surql
DEFINE TABLE authored SCHEMAFULL TYPE RELATION IN user OUT post ENFORCED;
DEFINE FIELD at ON authored TYPE datetime VALUE time::now();
DEFINE INDEX authored_unique ON authored FIELDS in, out UNIQUE;

RELATE user:jane->authored->post:123 SET at = time::now();
RELATE user:jane->follows->user:bob;

-- traversal
SELECT ->authored->post.*       AS posts   FROM user:jane;   -- forward
SELECT <-authored<-user         AS author  FROM post:123;    -- reverse
SELECT ->follows->user->authored->post AS feed FROM user:jane; -- multi-hop
SELECT ->reacted[WHERE kind = 'like']->post   FROM user:jane;  -- filtered edge
```

> Mental model: link = a pointer, array = a short list of pointers, edge = a thing in its own right. Properties on the relationship or a need to traverse it ⇒ edge. A junction table with two `record<T>` fields is the anti-pattern — use an edge. An edge for a plain 1:N is overkill — use a link.

### 7d. Graph analytics — "users who did X also did Y"

Pure-graph collaborative filtering: traverse out, traverse back, aggregate the overlap. Zero application logic.

```surql
DEFINE FUNCTION fn::recommend($user: record<user>, $limit: int) {
    LET $seen    = SELECT VALUE ->visited->business FROM ONLY $user;            -- what they know
    LET $similar = array::flatten(SELECT VALUE <-visited<-user FROM business WHERE id IN $seen);
    LET $ranked  = (
        SELECT out AS business, count() AS overlap
        FROM visited
        WHERE in IN $similar AND in != $user AND out NOT IN $seen
        GROUP BY out ORDER BY overlap DESC LIMIT $limit
    );
    RETURN SELECT * FROM business WHERE id IN $ranked.business;
};
```

### 7e. Materialized hierarchy path (link + `VALUE`, not a sub-query)

A stored breadcrumb that dereferences the parent *link* — recomputed each write, indexable, and allowed under the no-sub-query-in-`COMPUTED` rule because it reads a link, not a query result:

```surql
DEFINE FIELD parent ON category TYPE option<record<category>>;
DEFINE FIELD path   ON category VALUE
    IF parent != NONE THEN parent.path + ' > ' + name ELSE name END;
```

---

## 8. DEFINE EVENT

```surql
DEFINE EVENT [OVERWRITE | IF NOT EXISTS] name ON [TABLE] table
  WHEN condition
  THEN expression-or-block;
```

In-scope variables: `$event` (`'CREATE'|'UPDATE'|'DELETE'`), `$before`, `$after`, `$value`, `$this`.

```surql
-- a single expression is fine
DEFINE EVENT on_signup ON user WHEN $event = 'CREATE'
  THEN fn::send_welcome($after.id);

-- so is a multi-statement block — LET / IF / UPDATE / CREATE inline
DEFINE EVENT on_review_created ON review WHEN $event = 'CREATE' THEN {
    LET $biz   = $after.out;
    LET $stats = (SELECT math::mean(rating) AS avg, count() AS n FROM review WHERE out = $biz GROUP ALL);
    UPDATE $biz SET rating = $stats[0].avg OR 0, review_count = $stats[0].n OR 0;
    LET $owner = (SELECT VALUE in FROM manages WHERE out = $biz LIMIT 1)[0];
    IF $owner != NONE {
        CREATE activity SET type = 'review.received', target_owner = $owner, occurred_at = time::now();
    };
};

-- maintain an aggregate. Counts/sums: increment in place. Averages: recompute (a mean can't be nudged).
DEFINE FIELD comment_count ON post TYPE int DEFAULT 0;
DEFINE EVENT bump   ON comment WHEN $event = 'CREATE' THEN UPDATE $after.post  SET comment_count += 1;  -- count → increment
DEFINE EVENT unbump ON comment WHEN $event = 'DELETE' THEN UPDATE $before.post SET comment_count -= 1;
DEFINE EVENT rerate ON review WHEN $event = 'DELETE'                                                    -- mean → recompute
  THEN UPDATE $before.out SET rating = (SELECT VALUE math::mean(rating) FROM review WHERE out = $before.out GROUP ALL)[0] OR 0;
```

> Extract logic into a `fn::` only when it is reused across events or genuinely complex — a self-contained side effect is fine inline. Never write an event that `UPDATE`s its own table in a way that re-fires it — the recursive chain is silently killed. Fold `updated_at` into the triggering write instead.

---

## 9. DEFINE FUNCTION

```surql
DEFINE FUNCTION fn::name($arg: type, $arg2: type) {
    LET $x = …;
    IF $cond { THROW 'message' };
    RETURN value;
} [PERMISSIONS …];
```

```surql
-- guarded mutation (uniqueness still enforced by the UNIQUE edge index, not here)
DEFINE FUNCTION fn::follow($from: record<user>, $to: record<user>) {
    IF $from->follows->$to { RETURN false };
    RELATE $from->follows->$to;
    RETURN true;
};

-- reusable read / query-time derivation
DEFINE FUNCTION fn::price_in($amount: decimal, $code: string) -> decimal {
    LET $rate = (SELECT VALUE rate FROM exchange_rate WHERE code = $code)[0];
    RETURN $amount * ($rate ?? 1dec);
};
```

---

## 10. DEFINE SEQUENCE (human-readable codes)

```surql
DEFINE SEQUENCE [OVERWRITE | IF NOT EXISTS] name [BATCH n] [START n] [TIMEOUT duration];

DEFINE SEQUENCE invoice_no BATCH 100 START 1000;
CREATE invoice SET number = sequence::nextval('invoice_no');   -- 1000, 1001, …
```

Monotonic and globally unique across nodes. Use for order/invoice/reference numbers — never hand-increment in application code.

---

## 11. Authentication & authorization

### 11a. Record access (application end-users) — the auth method

```surql
DEFINE ACCESS account ON DATABASE TYPE RECORD
  SIGNUP (
    CREATE user CONTENT {
      email: $email,
      pass:  crypto::argon2::generate($pass),
      role:  'customer'
    }
  )
  SIGNIN (
    SELECT * FROM user
    WHERE email = $email
      AND crypto::argon2::compare(pass, $pass)
  )
  AUTHENTICATE {
    IF $auth.suspended { THROW 'Account suspended' };
    RETURN $auth;
  }
  WITH REFRESH
  DURATION FOR TOKEN 15m, FOR SESSION 12h;
```

- `crypto::argon2::generate(pass)` to hash, `crypto::argon2::compare(hash, pass)` to verify. (`crypto::bcrypt`, `crypto::scrypt`, `crypto::pbkdf2` also available; argon2 is the default choice.)
- `WITH REFRESH` returns an opaque, auto-rotating, revocable refresh token (HTTP REST only — do not promise it over WebSocket/RPC).
- `AUTHENTICATE` runs once at auth time (cheap place for claim checks / disabling / attempt logging); return the auth record or `THROW`.
- `DURATION FOR TOKEN …, FOR SESSION …` — keep the token short.
- Optional external/custom tokens: `… WITH JWT ALGORITHM HS512 KEY '…' …`.
- **No `sessions` table** — the engine issues and validates tokens.

### 11b. System users (administrators / operators) — separate concept

```surql
DEFINE USER admin ON ROOT     PASSWORD 'change-me'        ROLES OWNER;
DEFINE USER svc   ON DATABASE PASSHASH '$argon2id$...'    ROLES EDITOR;   -- OWNER | EDITOR | VIEWER
```

### 11c. Authorization — row/field level, in the schema

```surql
DEFINE TABLE post SCHEMAFULL PERMISSIONS
  FOR select WHERE published = true OR author = $auth.id
  FOR create WHERE $auth.id != NONE
  FOR update, delete WHERE author = $auth.id;

DEFINE FIELD internal_notes ON post TYPE option<string>
  PERMISSIONS FOR select WHERE $auth.role = 'admin';
```

Session params available in permissions/queries: `$auth` (the record user's record), `$token` (token claims), `$session` (session metadata), `$access` (access-method name). Always pass untrusted input as **bound parameters** (`$email`, `$pass`), never string-interpolated.

---

## 12. Derived-data recipes (decide by where inputs live)

```surql
-- A) pure function of the same record -> COMPUTED (read-time, never stale)
DEFINE FIELD total ON order_line COMPUTED unit_price * quantity;

-- B) aggregate of a record's own children -> stored field + EVENT (indexable)
DEFINE FIELD item_count ON cart TYPE int DEFAULT 0;
DEFINE EVENT inc ON cart_item WHEN $event = 'CREATE' THEN UPDATE $after.cart  SET item_count += 1;
DEFINE EVENT dec ON cart_item WHEN $event = 'DELETE' THEN UPDATE $before.cart SET item_count -= 1;

-- C) value from a separate volatile table, PINNED to a moment -> snapshot at write
DEFINE FIELD unit_price ON order_line TYPE decimal READONLY;   -- copied from product on CREATE

-- C') same, but LIVE for display -> query-time function, NOT a stored/computed field
SELECT *, fn::price_in(total, 'EUR') AS total_eur FROM order;
```

> Never bury a cross-table sub-query inside `COMPUTED` (per-read cost, unindexable, evaluation-order traps). If you must filter/sort on the value, snapshot the input so it becomes case A.

---

## 13. CRUD & query

```surql
-- create
CREATE user:jane SET name = 'Jane', email = 'jane@x.io';
CREATE user CONTENT { name: 'Jane', email: 'jane@x.io' };
INSERT INTO user [ { name: 'A' }, { name: 'B' } ];

-- upsert (create or update by id)
UPSERT user:jane SET visits += 1;   -- NB: += is order-dependent, so this WRITE is not idempotent (§24)

-- update
UPDATE user:jane MERGE { profile: { theme: 'dark' } };   -- deep-merge object
UPDATE user:jane SET tags += 'vip';                       -- append
UPDATE user:jane UNSET deprecated;                        -- remove a field
UPDATE user:jane PATCH [{ op: 'replace', path: '/name', value: 'J' }];

-- delete
DELETE user:jane;
DELETE user WHERE last_seen < time::now() - 1y;

-- select
SELECT name, email FROM user
  WHERE age >= 18 AND email = $email           -- bound param
  ORDER BY name ASC
  LIMIT 20 START 0;

SELECT count() AS total, role FROM user GROUP BY role;     -- aggregate per group
SELECT math::mean(score) AS avg, count() AS n FROM game GROUP ALL;  -- whole result in one row
SELECT * FROM ONLY user:jane;                              -- ONLY = single record, not an array
SELECT * OMIT pass FROM user;                              -- exclude fields
SELECT *, ->authored->post.* AS posts FROM user FETCH posts;
RETURN (SELECT VALUE email FROM user:jane);                -- VALUE = unwrap single field

-- optional filters in a parameterized query: pass NONE to skip a clause
SELECT * FROM business
  WHERE ($city = NONE OR city = $city)
    AND ($category = NONE OR category = $category);
```

---

## 14. Full-text search

```surql
DEFINE ANALYZER text TOKENIZERS class FILTERS lowercase, ascii, snowball(english);
DEFINE INDEX post_title_fts ON post FIELDS title FULLTEXT ANALYZER text BM25 HIGHLIGHTS;

SELECT
  title,
  search::score(0)                          AS score,
  search::highlight('<b>', '</b>', 0)       AS snippet
FROM post
WHERE title @0@ 'graph database'            -- @N@ binds to search::score(N)/highlight(…,N)
ORDER BY score DESC
LIMIT 10;
```

> One field per FTS index. `@@` is the match operator; the numbered form `@0@` links a match to its score/highlight. The pre-3.0 `SEARCH ANALYZER` spelling is invalid in v3.

---

## 15. Realtime — live queries (push) & changefeeds (pull)

```surql
-- PUSH: subscribe to changes as they happen
LIVE SELECT * FROM order WHERE status = 'pending';   -- pushes created/updated/deleted rows
LIVE SELECT DIFF FROM order;                          -- pushes JSON-patch diffs
KILL $live_query_id;                                  -- stop a live query

-- PULL: replay history / catch up / audit / sync to an external system
DEFINE TABLE reading CHANGEFEED 3d;                   -- opt the table in; retain 3 days of writes
DEFINE TABLE reading CHANGEFEED 3d INCLUDE ORIGINAL;  -- also store reverse diffs (prior state)
SHOW CHANGES FOR TABLE reading SINCE d'2026-01-01T00:00:00Z' LIMIT 100;  -- since a datetime
SHOW CHANGES FOR TABLE reading SINCE 1 LIMIT 100;     -- since a versionstamp cursor (page forward)
```

> Push vs pull: `LIVE SELECT` is a subscription for clients that must react now; a `CHANGEFEED` is a durable log a consumer pulls *after the fact* (`SHOW CHANGES SINCE <cursor>`) for replay, audit, or sync. Each `SHOW` batch returns entries carrying `changes` + a `versionstamp` — use that as the next cursor. Don't poll a live query for replay, and don't hand-roll an audit table for what the changefeed already captures.

---

## 16. Transactions & control flow

```surql
BEGIN TRANSACTION;
  LET $u = (CREATE ONLY user CONTENT { name: 'A' });
  LET $o = (CREATE ONLY order CONTENT { total: 10dec });
  RELATE $u->placed->$o;
COMMIT TRANSACTION;          -- or: CANCEL TRANSACTION;

LET $x = 10;
IF $x > 5 { RETURN 'big' } ELSE { RETURN 'small' };
FOR $row IN (SELECT * FROM staging) { CREATE record CONTENT $row };
THROW 'explicit error';      -- aborts and rolls back the surrounding transaction
```

---

## 17. Idempotent definitions (re-runnable init)

```surql
DEFINE TABLE OVERWRITE user SCHEMAFULL …;     -- always converge to declared state
DEFINE FIELD IF NOT EXISTS email ON user TYPE string;
DEFINE INDEX OVERWRITE user_email ON user FIELDS email UNIQUE;
```

Re-running the full pipeline (tables → fields → indexes → functions → events → seed) must be safe and converge. No migration chain; if a field tightens after data exists, edit it in place and run a one-shot backfill, then re-apply.

---

## 18. Common functions (cheat sheet)

```surql
time::now()            time::group(d'…', 'day')      duration::secs(1h)
rand::ulid()           rand::uuid()                   rand::int(1, 100)
type::thing('user',$id)  type::is::record($v)         record::id(user:jane)
string::is_email($v)   string::concat(a, ' ', b)      string::lowercase($v)   string::slug($v)
string::len($v)        string::split($v, ',')         string::join(', ', $arr)
array::len($a)         array::group($a)               array::distinct($a)     array::flatten($a)
array::all($a, |$x| …) array::any($a, |$x| …)         array::map($a, |$x| …)  array::filter($a, |$x| …)
geo::distance(a, b)    geo::area($poly)               geo::centroid($poly)    -- distance in metres
math::sum($a)          math::mean($a)                 math::round($n)         math::max($a)
count()                count($condition)              ->edge->node            <-edge<-node
crypto::argon2::generate($p)   crypto::argon2::compare($hash, $p)
search::score(0)       search::highlight('<b>','</b>',0)
sequence::nextval('seq_name')
$v.expect(|$x| cond, 'msg')    $v.chain(|$x| …)    -- v3.1+: assert-on / transform a value mid-chain (debug; clones)
```

---

## 19. HTTP `/sql` API (the built-in backend)

The engine exposes a fixed HTTP surface — consume it directly, do not build a wrapper backend. Reference: `github.com/surrealdb/openapi`.

```
POST /sql          body: raw SurrealQL
                   headers: Surreal-NS: <ns>, Surreal-DB: <db>,
                            Authorization: Bearer <jwt>, Accept: application/json
POST /signup       body: { ns, db, ac: 'account', <signup vars> }  -> { token, refresh? }
POST /signin       body: { ns, db, ac: 'account', email, pass }    -> { token }
GET|POST|PUT|PATCH|DELETE  /key/:table[/:id]    -- REST CRUD shortcuts
GET  /health   GET /version   GET /status
POST /import   POST /export   POST /graphql     POST /rpc
```

```bash
curl -X POST http://localhost:8000/sql \
  -H 'Surreal-NS: app' -H 'Surreal-DB: main' \
  -H 'Authorization: Bearer '"$TOKEN" -H 'Accept: application/json' \
  --data-binary 'SELECT * FROM user WHERE id = $auth.id;'
```

Use `DEFINE API "/path" FOR get MIDDLEWARE … THEN { … }` only to expose locked-down custom endpoints.

---

## 20. Capabilities (deploy posture — engine-enforced)

Secure by default: scripting, network, and arbitrary-query are denied unless enabled. Resolution: specific beats general beats global; at equal specificity, deny beats allow.

```bash
surreal start --deny-all \
  --allow-funcs "crypto::argon2,string,math,time,type,array,count,search,sequence" \
  --allow-guests          # ONLY if a public read surface is intended
# scripting & network stay denied (default). Add --allow-net host:443 only if an http::* call is required.
# For a public surface, also: --deny-arbitrary-query (for guest/record groups) + DEFINE API endpoints.
```

A `--allow-guests` guest can still only do what a resource's `PERMISSIONS` clause permits — `PERMISSIONS NONE` tables are invisible to them.

---

## 21. Internationalization (i18n)

Display strings live in a dedicated `locale_string` table; entities never carry per-language columns.

```surql
DEFINE TABLE locale_string SCHEMAFULL PERMISSIONS FOR select FULL;
DEFINE FIELD key    ON locale_string TYPE string;
DEFINE FIELD locale ON locale_string TYPE string;
DEFINE FIELD value  ON locale_string TYPE string;
DEFINE INDEX locale_unique ON locale_string FIELDS key, locale UNIQUE;

DEFINE FIELD locale ON user TYPE string DEFAULT 'en';   -- a record's preferred language

-- resolver with default-locale fallback
DEFINE FUNCTION fn::t($key: string, $locale: string) {
    LET $hit = (SELECT VALUE value FROM locale_string WHERE key = $key AND locale = $locale LIMIT 1);
    IF array::len($hit) > 0 { RETURN $hit[0] };
    LET $fb = (SELECT VALUE value FROM locale_string WHERE key = $key AND locale = 'en' LIMIT 1);
    RETURN array::first($fb) OR NONE;
};

RETURN fn::t('category_lodging', 'ja');
```

---

## 22. Geospatial

`geometry<point>` fields + `geo::*` functions. `geo::distance` returns **metres**.

```surql
DEFINE FIELD coordinates ON business TYPE geometry<point>;
DEFINE INDEX business_geo ON business FIELDS coordinates;

-- radius search, nearest first (augment each row with a computed distance)
DEFINE FUNCTION fn::near($point: geometry<point>, $radius_km: decimal) {
    LET $m = $radius_km * 1000.0dec;
    RETURN SELECT *, geo::distance(coordinates, $point) / 1000.0 AS distance_km
        FROM business
        WHERE is_active = true AND geo::distance(coordinates, $point) <= $m
        ORDER BY distance_km ASC;
};

-- point-in-polygon
SELECT * FROM business WHERE coordinates INSIDE $area;
```

---

## 23. Closures

`|$arg| expression` — first-class, passed to array functions and stored in fields/params.

```surql
array::all($langs,   |$l| $l IN ['es', 'en', 'fr'])       -- every element passes  (great in ASSERT)
array::any($scores,  |$s| $s >= 5)                          -- at least one passes
array::map($prices,  |$p| $p * 1.16dec)                     -- transform
array::filter($items,|$i| $i.active = true)                 -- keep matching
```

---

## 24. Idempotency & error handling

**Idempotent writes** — safe to retry, redeliver, or replay without corrupting data:

```surql
UPSERT user:123 SET name = 'Alice', status = 'active';   -- create-or-update by id (replaces IF-exists-THEN-update-ELSE-create)
DELETE user:123;                                          -- already idempotent: a second run is a no-op
DEFINE INDEX only_one ON likes FIELDS in, out UNIQUE;     -- makes RELATE idempotent: at most one edge per pair
RELATE user:123->likes->post:456;                         -- now safe to re-run

-- NOT idempotent — order-dependent accumulators; keep these off any retry-able path
UPDATE user:123 SET login_count += 1;
UPDATE user:123 SET tags += 'new';
```

> On a table with a `UNIQUE` index, `UPSERT` resolves the record through the index (no table scan), so it also beats `UPDATE` for single-record writes. Counters that must survive retries belong in an event-maintained aggregate (§8, §12-B), not a bare `+=`.

**Error-handling hierarchy** — push the rule as far down the stack as it goes:

```surql
-- 1. ASSERT (schema, permanent) — engine enforces on every write; the default home for an invariant
DEFINE FIELD email ON user TYPE string ASSERT string::is_email($value);

-- 2. THROW (query/transaction, runtime) — for guards a static ASSERT can't express; aborts + rolls back
BEGIN TRANSACTION;
  UPDATE account:one SET balance -= 150;
  IF account:one.balance < 0 { THROW 'Insufficient funds' };   -- value may be a string or a structured object
COMMIT TRANSACTION;
THROW { code: 400, message: 'Invalid request' };               -- structured error returned to the client

-- 3. .expect() (debugging, temporary, v3.1+) — checks a value mid-chain, returns it or fails with a message
LET $rows = SELECT * FROM person WHERE city = 'London';
$rows.expect(|$n| $n.len() > 0, 'Expected at least one Londoner').map(|$p| { name: $p.name });
```

> `.expect()` clones the value it checks — fine for a CLI/Surrealist spot-check or a `THROW`-terminated test transaction, but remove it from hot paths. For anything permanent, move the rule up to a `DEFINE FIELD … ASSERT`.

---

## 25. v3 gotchas (the traps that cause wrong output)

- **Field computation order is alphabetical** — name a dependent computed field to sort after its inputs.
- **Default referential action is `IGNORE`** — always state `ON DELETE REJECT|CASCADE|UNSET` explicitly.
- **No sub-query inside `COMPUTED`** — runs per-read per-record, can't be indexed; snapshot or use a `fn::` instead.
- **Events that update their own table** re-fire and the chain is silently dropped — fold side effects into the triggering write.
- **One field per full-text index**; `FULLTEXT ANALYZER` is the v3 spelling (not `SEARCH ANALYZER`).
- **Refresh tokens are HTTP REST only** — never model/promise refresh over WebSocket/RPC.
- **`DEFINE SCOPE` / `DEFINE TOKEN` were removed in 3.0** — use `DEFINE ACCESS`. There are no UUID/serial primary keys — identity is the record id.
- **`SCHEMALESS`, `FLEXIBLE`, `TYPE any`, untyped `object`** are not the default — express variability with literal-union types or wildcard-typed sub-fields.
- **`VALUE` vs `DEFAULT`** — `VALUE` runs on every write (use for `updated_at`); plain `DEFAULT` only on create (a `DEFAULT`-only `updated_at` never refreshes).
- **Record ranges need the partition prefix pinned** — `t:[k, lo]..=[k, hi]` is a fast partition scan; a `WHERE id[0] = k AND id[1] > lo …` over the same array ID falls back to a full table scan. `..=` includes the upper bound, `..` excludes it. Range bounds must share the composite-ID shape.
