#!/bin/bash
# Automatically create Apache AGE extension and set search_path for graph queries.
# Runs on first container init (docker-entrypoint-initdb.d).
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "postgres" <<-EOSQL
  CREATE EXTENSION IF NOT EXISTS age;
  ALTER DATABASE postgres SET search_path = ag_catalog, "\$user", public;
EOSQL

if [ -n "$POSTGRES_DB" ] && [ "$POSTGRES_DB" != "postgres" ]; then
  psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE EXTENSION IF NOT EXISTS age;
    ALTER DATABASE "$POSTGRES_DB" SET search_path = ag_catalog, "\$user", public;
EOSQL
fi
