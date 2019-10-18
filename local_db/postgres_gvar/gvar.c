/* Author: William Jay (wjay@fnal.gov)
 * Adapted from 
 * https://github.com/postgres/postgres/blob/master/src/tutorial/complex.c
 * 
 *
 ******************************************************************************
  This file contains routines that can be bound to a Postgres backend and
  called by the backend in the process of processing queries.  The calling
  format for these routines is dictated by Postgres architecture.
******************************************************************************/

#include "postgres.h"

#include "fmgr.h"
#include "libpq/pqformat.h"  /* needed for send/recv functions */
#include "math.h"            /* needed for sqrt */

PG_MODULE_MAGIC;

typedef struct Gvar
{
	double mean;
	double sdev;
}			Gvar;


/*****************************************************************************
 * Input/Output functions
 *****************************************************************************/

PG_FUNCTION_INFO_V1(gvar_in);

Datum
gvar_in(PG_FUNCTION_ARGS)
{
	char *str = PG_GETARG_CSTRING(0);
	double mean, sdev;
	Gvar *result;

	if (sscanf(str, " ( %lf , %lf )", &mean, &sdev) != 2)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
				 errmsg("invalid input syntax for type %s: \"%s\"",
						"gvar", str)));

	result = (Gvar *) palloc(sizeof(Gvar));
	result->mean = mean;
	result->sdev = sdev;
	PG_RETURN_POINTER(result);
}

PG_FUNCTION_INFO_V1(gvar_out);

Datum
gvar_out(PG_FUNCTION_ARGS)
{
	Gvar *gvar = (Gvar *) PG_GETARG_POINTER(0);
	char *result;

	result = psprintf("(%g,%g)", gvar->mean, gvar->sdev);
	PG_RETURN_CSTRING(result);
}

/*****************************************************************************
 * Binary Input/Output functions
 *
 * These are optional.
 *****************************************************************************/

PG_FUNCTION_INFO_V1(gvar_recv);

Datum
gvar_recv(PG_FUNCTION_ARGS)
{
	StringInfo buf = (StringInfo) PG_GETARG_POINTER(0);
	Gvar *result;

	result = (Gvar *) palloc(sizeof(Gvar));
	result->mean = pq_getmsgfloat8(buf);
	result->sdev = pq_getmsgfloat8(buf);
	PG_RETURN_POINTER(result);
}

PG_FUNCTION_INFO_V1(gvar_send);

Datum
gvar_send(PG_FUNCTION_ARGS)
{
	Gvar    *gvar = (Gvar *) PG_GETARG_POINTER(0);
	StringInfoData buf;

	pq_begintypsend(&buf);
	pq_sendfloat8(&buf, gvar->mean);
	pq_sendfloat8(&buf, gvar->sdev);
	PG_RETURN_BYTEA_P(pq_endtypsend(&buf));
}

/*****************************************************************************
 * New Operators
 *
 * A practical Gvar datatype would provide much more than this, of course.
 *****************************************************************************/

PG_FUNCTION_INFO_V1(gvar_add);

Datum
gvar_add(PG_FUNCTION_ARGS)
{
	Gvar    *a = (Gvar *) PG_GETARG_POINTER(0);
	Gvar    *b = (Gvar *) PG_GETARG_POINTER(1);
	Gvar    *result;

	result = (Gvar *) palloc(sizeof(Gvar));
	result->mean = a->mean + b->mean;
	result->sdev = sqrt((a)->sdev*(a)->sdev + (b)->sdev*(b)->sdev);
	PG_RETURN_POINTER(result);
}

PG_FUNCTION_INFO_V1(gvar_subtract);

Datum
gvar_subtract(PG_FUNCTION_ARGS)
{
    Gvar    *a = (Gvar *) PG_GETARG_POINTER(0);
    Gvar    *b = (Gvar *) PG_GETARG_POINTER(1);
    Gvar    *result;

    result = (Gvar *) palloc(sizeof(Gvar));
    result->mean = a->mean - b->mean;
	result->sdev = sqrt((a)->sdev*(a)->sdev + (b)->sdev*(b)->sdev);
	PG_RETURN_POINTER(result);
}

PG_FUNCTION_INFO_V1(gvar_multiply);

Datum
gvar_multiply(PG_FUNCTION_ARGS)
{
    Gvar    *a = (Gvar *) PG_GETARG_POINTER(0);
    Gvar    *b = (Gvar *) PG_GETARG_POINTER(1);
    Gvar    *result;

    result = (Gvar *) palloc(sizeof(Gvar));
    result->mean = (a)->mean*(b)->mean;
	result->sdev = sqrt(
        (b)->mean*(b)->mean * (a)->sdev*(a)->sdev
        + (a)->mean*(a)->mean * (b)->sdev*(b)->sdev
    );
	PG_RETURN_POINTER(result);
}

PG_FUNCTION_INFO_V1(gvar_divide);

Datum
gvar_divide(PG_FUNCTION_ARGS)
{
    Gvar    *a = (Gvar *) PG_GETARG_POINTER(0);
    Gvar    *b = (Gvar *) PG_GETARG_POINTER(1);
    Gvar    *result;

    result = (Gvar *) palloc(sizeof(Gvar));
    result->mean = (a)->mean/(b)->mean;
	result->sdev = sqrt(
        (a)->sdev*(a)->sdev / ((b)->mean*(b)->mean)
        + (a)->mean*(a)->mean * (b)->sdev*(b)->sdev 
          / ((b)->mean*(b)->mean*(b)->mean*(b)->mean)
    );
	PG_RETURN_POINTER(result);
}

/*****************************************************************************
 * Operator class for defining B-tree index
 *
 * It's essential that the comparison operators and support function for a
 * B-tree index opclass always agree on the relative ordering of any two
 * data values.  Experience has shown that it's depressingly easy to write
 * unintentionally inconsistent functions.  One way to reduce the odds of
 * making a mistake is to make all the functions simple wrappers around
 * an internal three-way-comparison function, as we do here.
 * 
 * Ordering operators for gvars will simply use the mean, ignoring the error.
 *****************************************************************************/


static int
gvar_mean_cmp_internal(Gvar * a, Gvar * b)
{
	double		amean = a->mean,
				bmean = b->mean;

	if (amean < bmean)
		return -1;
	if (amean > bmean)
		return 1;
	return 0;
}


PG_FUNCTION_INFO_V1(gvar_mean_lt);

Datum
gvar_mean_lt(PG_FUNCTION_ARGS)
{
	Gvar    *a = (Gvar *) PG_GETARG_POINTER(0);
	Gvar    *b = (Gvar *) PG_GETARG_POINTER(1);

	PG_RETURN_BOOL(gvar_mean_cmp_internal(a, b) < 0);
}

PG_FUNCTION_INFO_V1(gvar_mean_le);

Datum
gvar_mean_le(PG_FUNCTION_ARGS)
{
	Gvar    *a = (Gvar *) PG_GETARG_POINTER(0);
	Gvar    *b = (Gvar *) PG_GETARG_POINTER(1);

	PG_RETURN_BOOL(gvar_mean_cmp_internal(a, b) <= 0);
}

PG_FUNCTION_INFO_V1(gvar_mean_eq);

Datum
gvar_mean_eq(PG_FUNCTION_ARGS)
{
	Gvar    *a = (Gvar *) PG_GETARG_POINTER(0);
	Gvar    *b = (Gvar *) PG_GETARG_POINTER(1);

	PG_RETURN_BOOL(gvar_mean_cmp_internal(a, b) == 0);
}

PG_FUNCTION_INFO_V1(gvar_mean_ge);

Datum
gvar_mean_ge(PG_FUNCTION_ARGS)
{
	Gvar    *a = (Gvar *) PG_GETARG_POINTER(0);
	Gvar    *b = (Gvar *) PG_GETARG_POINTER(1);

	PG_RETURN_BOOL(gvar_mean_cmp_internal(a, b) >= 0);
}

PG_FUNCTION_INFO_V1(gvar_mean_gt);

Datum
gvar_mean_gt(PG_FUNCTION_ARGS)
{
	Gvar    *a = (Gvar *) PG_GETARG_POINTER(0);
	Gvar    *b = (Gvar *) PG_GETARG_POINTER(1);

	PG_RETURN_BOOL(gvar_mean_cmp_internal(a, b) > 0);
}

PG_FUNCTION_INFO_V1(gvar_mean_cmp);

Datum
gvar_mean_cmp(PG_FUNCTION_ARGS)
{
	Gvar    *a = (Gvar *) PG_GETARG_POINTER(0);
	Gvar    *b = (Gvar *) PG_GETARG_POINTER(1);

	PG_RETURN_INT32(gvar_mean_cmp_internal(a, b));
}