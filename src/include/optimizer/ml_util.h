#ifndef ML_UTIL_H
#define ML_UTIL_H
#include <arpa/inet.h>
#include <unistd.h>
#include <string.h>
#include "utils/ruleutils.h"

static char* leon_host = "localhost";
static int leon_port = 9999;

// JSON tags for sending to the leon server.
static const char* START_QUERY_MESSAGE = "{\"type\": \"query\"}\n";
static const char *START_FEEDBACK_MESSAGE = "{\"type\": \"reward\"}\n";
static const char* START_PREDICTION_MESSAGE = "{\"type\": \"predict\"}\n";
static const char* TERMINAL_MESSAGE = "{\"final\": true}\n";

typedef struct
{
	List	   *rtable;			/* List of RangeTblEntry nodes */
	List	   *rtable_names;	/* Parallel list of names for RTEs */
	List	   *rtable_columns; /* Parallel list of deparse_columns structs */
	List	   *subplans;		/* List of Plan trees for SubPlans */
	List	   *ctes;			/* List of CommonTableExpr nodes */
	AppendRelInfo **appendrels; /* Array of AppendRelInfo nodes, or NULL */
	/* Workspace for column alias assignment: */
	bool		unique_using;	/* Are we making USING names globally unique */
	List	   *using_names;	/* List of assigned names for USING columns */
	/* Remaining fields are used only when deparsing a Plan tree: */
	Plan	   *plan;			/* immediate parent of current expression */
	List	   *ancestors;		/* ancestors of plan */
	Plan	   *outer_plan;		/* outer subnode, or NULL if none */
	Plan	   *inner_plan;		/* inner subnode, or NULL if none */
	List	   *outer_tlist;	/* referent for OUTER_VAR Vars */
	List	   *inner_tlist;	/* referent for INNER_VAR Vars */
	List	   *index_tlist;	/* referent for INDEX_VAR Vars */
	/* Special namespace representing a function signature: */
	char	   *funcname;
	int			numargs;
	char	  **argnames;
} deparse_namespace;

extern void set_simple_column_names(deparse_namespace *dpns);
extern char *deparse_expression_pretty(Node *expr, List *dpcontext,
									   bool forceprefix, bool showimplicit,
									   int prettyFlags, int startIndent);

static bool should_leon_optimize(int level) {
  return true;
}

static void get_calibrations(double calibrations[], uint32 queryid, int32_t length, int conn_fd){
  		// Read the response from the server and store it in the calibrations array
      // one element is like "1.12," length 5
      char *response = (char *)calloc(5 * length, sizeof(char));
      if (read(conn_fd, response, 5 * length * sizeof(char)) > 0) 
      {
        char *token = strtok(response, ",");
        int i = 0;
        while (token != NULL) 
        {
          calibrations[i] = atof(token);
          token = strtok(NULL, ",");
          i++;
        }
        Assert(i == length);
        if (i != length)
        {
          elog(ERROR, "Python code get wrong number of results!");
          exit(1);
        }
      } else {
        shutdown(conn_fd, SHUT_RDWR);
        elog(WARNING, "LEON could not read the response from the server.");
      }
    
      free(response);
}


static int connect_to_leon(const char* host, int port) {
  int ret, conn_fd;
  struct sockaddr_in server_addr = { 0 };

  server_addr.sin_family = AF_INET;
  server_addr.sin_port = htons(port);
  inet_pton(AF_INET, host, &server_addr.sin_addr);
  conn_fd = socket(AF_INET, SOCK_STREAM, 0);
  if (conn_fd < 0) {
    return conn_fd;
  }
  
  ret = connect(conn_fd, (struct sockaddr*)&server_addr, sizeof(server_addr));
  if (ret == -1) {
    return ret;
  }

  return conn_fd;

}


static void write_all_to_socket(int conn_fd, const char* json) {
  size_t json_length;
  ssize_t written, written_total;
  json_length = strlen(json);
  written_total = 0;
  
  while (written_total != json_length) {
    written = write(conn_fd,
                    json + written_total,
                    json_length - written_total);
    written_total += written;
  }
}

List *
deparse_context_for_path(PlannerInfo *root, List *rtable_names)
{
	deparse_namespace *dpns;

	dpns = (deparse_namespace *) palloc0(sizeof(deparse_namespace));

	/* Initialize fields that stay the same across the whole plan tree */
	dpns->rtable = root->parse->rtable; 
	dpns->rtable_names = rtable_names;
	dpns->subplans = root->glob->subplans;
	dpns->ctes = NIL;
	if (root->glob->appendRelations)
	{
		/* Set up the array, indexed by child relid */
		int			ntables = list_length(dpns->rtable);
		ListCell   *lc;

		dpns->appendrels = (AppendRelInfo **)
			palloc0((ntables + 1) * sizeof(AppendRelInfo *));
		foreach(lc, root->glob->appendRelations)
		{
			AppendRelInfo *appinfo = lfirst_node(AppendRelInfo, lc);
			Index		crelid = appinfo->child_relid;

			Assert(crelid > 0 && crelid <= ntables);
			Assert(dpns->appendrels[crelid] == NULL);
			dpns->appendrels[crelid] = appinfo;
		}
	}
	else
		dpns->appendrels = NULL;	/* don't need it */

	/*
	 * Set up column name aliases.  We will get rather bogus results for join
	 * RTEs, but that doesn't matter because plan trees don't contain any join
	 * alias Vars.
	 */
	set_simple_column_names(dpns);

	/* Return a one-deep namespace stack */
	return list_make1(dpns);
}

static void
debug_print_relids(PlannerInfo *root, Relids relids, FILE* stream)
{
	int			x;
	bool		first = true;

	x = -1;
	while ((x = bms_next_member(relids, x)) >= 0)
	{
		if (!first)
			fprintf(stream, " ");
		if (x < root->simple_rel_array_size &&
			root->simple_rte_array[x])
			fprintf(stream, "%s", root->simple_rte_array[x]->eref->aliasname);
		else
			fprintf(stream, "%d", x);
		first = false;
	}
}

void
debug_print_joincond(PlannerInfo *root, RelOptInfo *rel, File* stream)
{
	ListCell   *lc;
	List *rtable = root->parse->rtable;

	if (rel->reloptkind != RELOPT_JOINREL)
		return;

	bool first = true;
	foreach(lc, root->parse->jointree->quals)
	{
		Node *expr = (Node *) lfirst(lc);
		if IsA(expr, OpExpr)
		{	
			const OpExpr *e = (const OpExpr *) expr;
			char	   *opname;

			opname = get_opname(e->opno);
			if (list_length(e->args) > 1)
			{	
				Node * left_node = get_leftop((const Expr *) e);
				Node * right_node = get_rightop((const Expr *) e);
				if (IsA(left_node, Var) && IsA(right_node, Var))
				{	
					Var *left_var = (Var *) left_node;
					Var *right_var = (Var *) right_node;
					//Both vars are from the same relation
					if (bms_is_member(left_var->varno, rel->relids) &&
						bms_is_member(right_var->varno, rel->relids))
					{	
						if (!first)
							fprintf(stream, ", ");
						debug_print_expr(left_node, rtable, stream);
						fprintf(stream, " %s ", ((opname != NULL) ? opname : "(invalid operator)"));
						debug_print_expr(right_node, rtable, stream);
						first = false;
					}
				}
			}		
		}
	}
}

/*
 * debug_print_expr
 *	  print an expression to a file
 */
void
debug_print_expr(const Node *expr, const List *rtable, FILE* stream)
{
	if (expr == NULL)
	{
		fprintf(stream, "<>");
		return;
	}

	if (IsA(expr, Var))
	{
		const Var  *var = (const Var *) expr;
		char	   *relname,
				   *attname;

		switch (var->varno)
		{
			case INNER_VAR:
				relname = "INNER";
				attname = "?";
				break;
			case OUTER_VAR:
				relname = "OUTER";
				attname = "?";
				break;
			case INDEX_VAR:
				relname = "INDEX";
				attname = "?";
				break;
			default:
				{
					RangeTblEntry *rte;

					Assert(var->varno > 0 &&
						   (int) var->varno <= list_length(rtable));
					rte = rt_fetch(var->varno, rtable);
					relname = rte->eref->aliasname;
					attname = get_rte_attribute_name(rte, var->varattno);
				}
				break;
		}
		fprintf(stream, "%s.%s", relname, attname);
	}
	else if (IsA(expr, Const))
	{
		const Const *c = (const Const *) expr;
		Oid			typoutput;
		bool		typIsVarlena;
		char	   *outputstr;

		if (c->constisnull)
		{
			fprintf(stream, "NULL");
			return;
		}

		getTypeOutputInfo(c->consttype,
						  &typoutput, &typIsVarlena);

		outputstr = OidOutputFunctionCall(typoutput, c->constvalue);
		fprintf(stream, "\'%s\'", outputstr);
		pfree(outputstr);
	}
	else if (IsA(expr, OpExpr))
	{
		const OpExpr *e = (const OpExpr *) expr;
		char	   *opname;

		opname = get_opname(e->opno);
		if (list_length(e->args) > 1)
		{
			debug_print_expr(get_leftop((const Expr *) e), rtable,  stream);
			fprintf(stream, " %s ", ((opname != NULL) ? opname : "(invalid operator)"));
			debug_print_expr(get_rightop((const Expr *) e), rtable, stream);
		}
		else
		{
			fprintf(stream, "%s ", ((opname != NULL) ? opname : "(invalid operator)"));
			debug_print_expr(get_leftop((const Expr *) e), rtable, stream);
		}
	}
	else if (IsA(expr, FuncExpr))
	{
		const FuncExpr *e = (const FuncExpr *) expr;
		char	   *funcname;
		ListCell   *l;

		funcname = get_func_name(e->funcid);
		fprintf(stream, "%s(", ((funcname != NULL) ? funcname : "(invalid function)"));
		foreach(l, e->args)
		{
			debug_print_expr(lfirst(l), rtable, stream);
			if (lnext(e->args, l))
				fprintf(stream, ",");
		}
		fprintf(stream, ")");
	}
	else if (IsA(expr, RelabelType))
 	{
		const RelabelType *r = (const RelabelType*) expr;

		debug_print_expr((Node *) r->arg, rtable, stream);
	}
	else if (IsA(expr, RangeTblRef))
	{
		int	varno = ((RangeTblRef *) expr)->rtindex;
		RangeTblEntry *rte = rt_fetch(varno, rtable);
		fprintf(stream, "RTE %d (%s)", varno, rte->eref->aliasname);
	}
	else
		fprintf(stream, "unknown expr");
}

List *
create_context(PlannerInfo *root)
{
	List *rtable_names = NIL;
	ListCell *lc;
	foreach(lc, root->parse->rtable)
	{
		RangeTblEntry *rte = lfirst(lc);
		rtable_names = lappend(rtable_names, rte->eref->aliasname);
	}
	List * context = deparse_context_for_path(root, rtable_names);
	return context;
}

void 
delete_context(List *context)
{
	ListCell *lc;
	foreach(lc, context)
	{
		deparse_namespace *dpns = lfirst(lc);
		pfree(dpns);
	}
	list_free(context);
}

static void
debug_print_restrictclauses(PlannerInfo *root, List *clauses, List *context, FILE* stream)
{
	ListCell   *l;
	foreach(l, clauses)
	{
		RestrictInfo *c = lfirst(l);
		// char * str = deparse_expression(c->clause, context, true, false);
		char * str = deparse_expression_pretty(c->clause, context, true,
									 false, true, 0);
		fprintf(stream, "%s", str);
		if (str)
			pfree(str);
		// pfree context
		if (lnext(clauses, l))
			fprintf(stream, ", ");
	}
}

static void
debug_print_path(PlannerInfo *root, Path *path, int indent, FILE* stream)
{
	const char *ptype;
	bool join = false;
	Path *subpath = NULL;
	int i;
	// StringInfoData buf;
	char *pathBufPtr = NULL;

	// initStringInfo(&buf);

	switch (nodeTag(path))
	{
		case T_Path:
			switch (path->pathtype)
			{
				case T_SeqScan:
					ptype = "SeqScan";
					break;
				case T_SampleScan:
					ptype = "SampleScan";
					break;
				case T_FunctionScan:
					ptype = "FunctionScan";
					break;
				case T_TableFuncScan:
					ptype = "TableFuncScan";
					break;
				case T_ValuesScan:
					ptype = "ValuesScan";
					break;
				case T_CteScan:
					ptype = "CteScan";
					break;
				case T_NamedTuplestoreScan:
					ptype = "NamedTuplestoreScan";
					break;
				case T_Result:
					ptype = "Result";
					break;
				case T_WorkTableScan:
					ptype = "WorkTableScan";
					break;
				default:
					ptype = "???Path";
					break;
			}
			break;
		case T_IndexPath:
			ptype = "IdxScan";
			break;
		case T_BitmapHeapPath:
			ptype = "BitmapHeapScan";
			break;
		case T_BitmapAndPath:
			ptype = "BitmapAndPath";
			break;
		case T_BitmapOrPath:
			ptype = "BitmapOrPath";
			break;
		case T_TidPath:
			ptype = "TidScan";
			break;
		case T_SubqueryScanPath:
			ptype = "SubqueryScan";
			break;
		case T_ForeignPath:
			ptype = "ForeignScan";
			break;
		case T_CustomPath:
			ptype = "CustomScan";
			break;
		case T_NestPath:
			ptype = "NestLoop";
			join = true;
			break;
		case T_MergePath:
			ptype = "MergeJoin";
			join = true;
			break;
		case T_HashPath:
			ptype = "HashJoin";
			join = true;
			break;
		case T_AppendPath:
			ptype = "Append";
			break;
		case T_MergeAppendPath:
			ptype = "MergeAppend";
			break;
		case T_GroupResultPath:
			ptype = "GroupResult";
			break;
		case T_MaterialPath:
			ptype = "Material";
			subpath = ((MaterialPath *) path)->subpath;
			break;
		case T_MemoizePath:
			ptype = "Memoize";
			subpath = ((MemoizePath *) path)->subpath;
			break;
		case T_UniquePath:
			ptype = "Unique";
			subpath = ((UniquePath *) path)->subpath;
			break;
		case T_GatherPath:
			ptype = "Gather";
			subpath = ((GatherPath *) path)->subpath;
			break;
		case T_GatherMergePath:
			ptype = "GatherMerge";
			subpath = ((GatherMergePath *) path)->subpath;
			break;
		case T_ProjectionPath:
			ptype = "Projection";
			subpath = ((ProjectionPath *) path)->subpath;
			break;
		case T_ProjectSetPath:
			ptype = "ProjectSet";
			subpath = ((ProjectSetPath *) path)->subpath;
			break;
		case T_SortPath:
			ptype = "Sort";
			subpath = ((SortPath *) path)->subpath;
			break;
		case T_IncrementalSortPath:
			ptype = "IncrementalSort";
			subpath = ((SortPath *) path)->subpath;
			break;
		case T_GroupPath:
			ptype = "Group";
			subpath = ((GroupPath *) path)->subpath;
			break;
		case T_UpperUniquePath:
			ptype = "UpperUnique";
			subpath = ((UpperUniquePath *) path)->subpath;
			break;
		case T_AggPath:
			ptype = "Agg";
			subpath = ((AggPath *) path)->subpath;
			break;
		case T_GroupingSetsPath:
			ptype = "GroupingSets";
			subpath = ((GroupingSetsPath *) path)->subpath;
			break;
		case T_MinMaxAggPath:
			ptype = "MinMaxAgg";
			break;
		case T_WindowAggPath:
			ptype = "WindowAgg";
			subpath = ((WindowAggPath *) path)->subpath;
			break;
		case T_SetOpPath:
			ptype = "SetOp";
			subpath = ((SetOpPath *) path)->subpath;
			break;
		case T_RecursiveUnionPath:
			ptype = "RecursiveUnion";
			break;
		case T_LockRowsPath:
			ptype = "LockRows";
			subpath = ((LockRowsPath *) path)->subpath;
			break;
		case T_ModifyTablePath:
			ptype = "ModifyTable";
			break;
		case T_LimitPath:
			ptype = "Limit";
			subpath = ((LimitPath *) path)->subpath;
			break;
		default:
			ptype = "???Path";
			break;
	}

  fprintf(stream, "{\"Node Type\": \"%s\",", ptype);
  fprintf(stream, "\"Node Type ID\": \"%d\",", path->type);
	if (path->parent)
	{
		fprintf(stream, "\"Relation IDs\": \"");
		debug_print_relids(root, path->parent->relids, stream);
		fprintf(stream, "\",");

		// Get context
		List *context = NIL;

		if (path->parent->baserestrictinfo)
		{	
			context = create_context(root);
			fprintf(stream, "\"Base Restrict Info\": \"");
			debug_print_restrictclauses(root, path->parent->baserestrictinfo, context, stream);
			fprintf(stream, "\",");
		}

		if (path->parent->joininfo)
		{	
			if (!context)
				context = create_context(root);
			fprintf(stream, "\"Join Info\": \"");
			debug_print_restrictclauses(root, path->parent->joininfo, context, stream);
			fprintf(stream, "\",");
		}
		if (context)
			delete_context(context);
	}
	if (path->param_info)
	{
    	fprintf(stream, "\"Required Outer\": \"");
		debug_print_relids(root, path->param_info->ppi_req_outer, stream);
		fprintf(stream, "\",");
	}
	if (path->parent->reloptkind == RELOPT_JOINREL)
	{	
		fprintf(stream, "\"Join Cond\": \"");
		debug_print_joincond(root, path->parent, stream);
		fprintf(stream, "\",");
	}
	if (path->pathtarget)
	{	
		fprintf(stream, "\"Path Target\": \"");
		PathTarget *pathtarget = path->pathtarget;
        ListCell *lc_expr;
		bool first = true;
        foreach(lc_expr, pathtarget->exprs) {
			if (!first)
				fprintf(stream, ", ");
            Node *expr = (Node *) lfirst(lc_expr);
            debug_print_expr(expr, root->parse->rtable, stream);
			first = false;
        }
		fprintf(stream, "\",");
	}

  fprintf(stream, "\"Startup Cost\": %f,", path->startup_cost);
  fprintf(stream, "\"Total Cost\": %f,", path->total_cost);
  fprintf(stream, "\"Plan Rows\": %f", path->rows);



	// if (path->pathkeys)
	// {
	// 	fprintf(stream, "\"Pathkeys\": ");
	// 	print_pathkeys(path->pathkeys, root->parse->rtable);
	// }

	if (join)
	{
		JoinPath *jp = (JoinPath *)path;

		// for (i = 0; i < indent; i++)
		// 	appendStringInfoString(&buf, "\t");
		// appendStringInfoString(&buf, "  clauses: ");
		// print_restrictclauses(root, jp->joinrestrictinfo);
		// appendStringInfoString(&buf, "\n");

		if (IsA(path, MergePath))
		{
			MergePath *mp = (MergePath *)path;

			fprintf(stream, ", \"Sort Outer\": %f,", ((mp->outersortkeys) ? 1 : 0));
			fprintf(stream, "\"Sort Inner\": %f,", ((mp->innersortkeys) ? 1 : 0));
			fprintf(stream, "\"Materialize Inner\": %f", ((mp->materialize_inner) ? 1 : 0));
		}

    fprintf(stream, ", \"Plans\": [");
		debug_print_path(root, jp->outerjoinpath, indent + 1, stream);
    fprintf(stream, ", ");
		debug_print_path(root, jp->innerjoinpath, indent + 1, stream);
    fprintf(stream, "]");
	}

	if (subpath)
	{ 
	// Plans or SubPlan
    fprintf(stream, ", \"Plans\": [");
		debug_print_path(root, subpath, indent + 1, stream);
		fprintf(stream, "]");
	}
	
  fprintf(stream, "}");
}



static char* plan_to_json(PlannerInfo * root, Path* plan) {
  char* buf;
  size_t json_size;
  FILE* stream;
  
  stream = open_memstream(&buf, &json_size);
  fprintf(stream, "{\"Plan\": ");
  debug_print_path(root, plan, 0, stream);
  fprintf(stream, "");
  fprintf(stream, "\"QueryId\": \"%d\",", root->parse->queryId);
  fprintf(stream,"}\n");
  fclose(stream);
  
  return buf;
}

#endif