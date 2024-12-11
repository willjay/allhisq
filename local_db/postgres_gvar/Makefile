#-------------------------------------------------------------------------
# Author: William Jay (wjay@fnal.gov)
# Adapted from 
# https://github.com/postgres/postgres/blob/master/src/tutorial/Makefile
# Makefile--
#    Makefile for tutorial
#
# By default, this builds against an existing PostgreSQL installation
# (the one identified by whichever pg_config is first in your path).
# Within a configured source tree, you can say "make NO_PGXS=1 all"
# to build using the surrounding source tree.
#
# IDENTIFICATION
#    src/tutorial/Makefile
#
#-------------------------------------------------------------------------

MODULES = gvar 
DATA_built = gvar.sql

ifdef NO_PGXS
subdir = src/tutorial
top_builddir = ../..
include $(top_builddir)/src/Makefile.global
include $(top_srcdir)/src/makefiles/pgxs.mk
else
PG_CONFIG = /Applications/Postgres.app/Contents/Versions/latest/bin/pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)
endif

%.sql: %.source
	rm -f $@; \
	C=`pwd`; \
	sed -e "s:_OBJWD_:$$C:g" < $< > $@