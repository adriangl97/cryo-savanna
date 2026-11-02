// Copyright (C) 2026 Adrián González-López
// SPDX-License-Identifier: GPL-2.0-or-later
// Derived from RELION (Sjors H.W. Scheres, MRC Laboratory of Molecular Biology; GPL-2.0-or-later): see NOTICE.md.
// RELION's notice, kept as it asks:
//   Author: "Sjors H.W. Scheres"
//   MRC Laboratory of Molecular Biology
//
//   This program is free software; you can redistribute it and/or modify
//   it under the terms of the GNU General Public License as published by
//   the Free Software Foundation; either version 2 of the License, or
//   (at your option) any later version.
//
//   This program is distributed in the hope that it will be useful,
//   but WITHOUT ANY WARRANTY; without even the implied warranty of
//   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//   GNU General Public License for more details.
//
//   This complete copyright notice must be included in any revised version of the
//   source code. Additional authorship citations may be added, but existing
//   author citations must be preserved.
// Reference implementation of the pipeline operations relion_pipeliner has no flags for.
//
// Calls RELION's own PipeLine methods the way the original GUI does (gui_mainwindow.cpp), so
// tests can compare cryo-savanna's interim Python versions with RELION's. Build with build.py
// against a RELION build directory.
//
//   ops_ref mark_finished JOB      ops_ref mark_failed JOB
//   ops_ref set_alias JOB ALIAS    (ALIAS "" removes the alias)
//   ops_ref delete JOB             ops_ref delete_recursive JOB
//   ops_ref undelete Trash/<Type>/<jobNNN>/deleted_pipeline.star

#include <iostream>
#include <string>
#include <vector>
#include "src/pipeliner.h"

static int fail(const std::string &msg)
{
	std::cerr << msg << std::endl;
	return 1;
}

int main(int argc, char **argv)
{
	if (argc < 3)
		return fail("usage: ops_ref OPERATION JOB [ARG]");
	std::string op = argv[1], target = argv[2];

	PipeLine pipeline;
	pipeline.read(DONT_LOCK);  // the GUI holds the pipeline in memory; each operation re-reads with the lock

	if (op == "undelete")
	{
		pipeline.undeleteJob(target);
		return 0;
	}

	long int job = pipeline.findProcessByName(target);
	if (job < 0)
		return fail("no such job: " + target);
	std::string error_message;

	if (op == "mark_finished" || op == "mark_failed")
	{
		if (!pipeline.markAsFinishedJob(job, error_message, op == "mark_failed"))
			return fail(error_message);
	}
	else if (op == "set_alias")
	{
		if (argc < 4)
			return fail("set_alias needs an alias");
		if (!pipeline.setAliasJob(job, argv[3], error_message))
			return fail(error_message);
	}
	else if (op == "delete" || op == "delete_recursive")
	{
		std::vector<bool> deleteProcesses, deleteNodes;
		pipeline.deleteJobGetNodesAndProcesses(job, op == "delete_recursive", deleteNodes, deleteProcesses);
		pipeline.deleteNodesAndProcesses(deleteNodes, deleteProcesses);
	}
	else
		return fail("unknown operation: " + op);
	return 0;
}
