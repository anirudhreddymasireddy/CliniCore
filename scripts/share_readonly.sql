-- Give other people READ-ONLY access to the CliniCore results (run as SYS in Exasol, e.g. in DBeaver or the Exasol worksheet).
-- Exasol Personal runs on your laptop: others can only reach it on your network. For access over the internet use an
-- Exasol SaaS database and add their IP addresses to its allow list (Security page). Never share the SYS password.
CREATE USER CLINICORE_READER IDENTIFIED BY "change-this-password";
GRANT CREATE SESSION TO CLINICORE_READER;
GRANT SELECT ON SCHEMA CAREGUARD_RESULTS TO CLINICORE_READER;     -- model outputs, ICU board, staff feedback, retraining log
GRANT SELECT ON SCHEMA CAREGUARD_SEPSIS19 TO CLINICORE_READER;    -- PhysioNet 2019 (open data, cite the challenge)
-- GRANT SELECT ON SCHEMA CAREGUARD_NHAMCS TO CLINICORE_READER;   -- NHAMCS ED (US public domain), if loaded with nhamcs.py --exasol
-- Do NOT grant access to any schema holding MIMIC-IV data.
