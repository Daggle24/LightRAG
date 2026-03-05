# PostgreSQL 18 with pgvector and Apache AGE extensions
FROM postgres:18

# Install build dependencies for pgvector and Apache AGE
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    git \
    postgresql-server-dev-18 \
    ca-certificates \
    libreadline-dev \
    zlib1g-dev \
    flex \
    bison && \
    # Clone and build pgvector (using main branch for PostgreSQL 18 compatibility)
    cd /tmp && \
    git clone https://github.com/pgvector/pgvector.git && \
    cd pgvector && \
    make && \
    make install && \
    cd /tmp && \
    rm -rf pgvector && \
    # Clone and build Apache AGE (graph extension for PostgreSQL)
    git clone https://github.com/apache/age.git && \
    cd age && \
    make install && \
    cd /tmp && \
    rm -rf age && \
    # Cleanup build dependencies
    cd / && \
    apt-get remove -y build-essential git postgresql-server-dev-18 libreadline-dev zlib1g-dev flex bison && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Add configuration for pgvector and Apache AGE (auto-load age in every session)
RUN echo "shared_preload_libraries = 'vector'" >> /usr/share/postgresql/postgresql.conf.sample && \
    echo "session_preload_libraries = 'age'" >> /usr/share/postgresql/postgresql.conf.sample

# Init script: create AGE extension and set search_path on first container start
COPY docker/postgres/02-init-age.sh /docker-entrypoint-initdb.d/
RUN chmod +x /docker-entrypoint-initdb.d/02-init-age.sh

# Expose PostgreSQL port
EXPOSE 5432

# Use the default postgres entrypoint
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["postgres"]
