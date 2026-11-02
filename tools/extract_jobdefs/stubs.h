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
//
//   The XMIPP code in RELION's src/strings.cpp (GPL-2.0-or-later; bestPrecision, floatToString,
//   textToInteger, textToFloat, textToBool):
//   Authors: J.R. Bilbao-Castro (jrbcast@ace.ual.es)
//   Unidad de Bioinformatica of Centro Nacional de Biotecnologia, CSIC
// Stand-ins for the RELION/FLTK classes used by the job-option and job-window initialisers.
//
// extract.py pastes the verbatim RelionJob::initialise* and JobWindow::initialise* functions
// from the RELION source after this header, so that compiling and running them records every
// job option (with its exact default, range, help) and the GUI layout (tabs, groups, toggles).
// The JobOption constructor overloads below must stay identical to RELION's so the compiler
// picks the same constructor for each call.
#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

typedef double RFLOAT;

struct RelionError
{
	std::string msg;
};
#define REPORT_ERROR(m) throw RelionError{std::string(m)}

// --- string helpers (copied semantics from src/strings.cpp) ---------------------------------
inline int bestPrecision(float F, int _width)
{
	if (F == 0) return 1;
	int exp = (int)std::floor(std::log10(std::fabs(F)));
	int advised_prec;
	if (exp >= 0)
		advised_prec = (exp > _width - 3) ? -1 : _width - 2;
	else
	{
		advised_prec = _width + (exp - 1) - 3;
		if (advised_prec <= 0) advised_prec = -1;
	}
	if (advised_prec < 0) advised_prec = -1;
	return advised_prec;
}

inline std::string floatToString(float F, int _width = 0, int _prec = 0)
{
	std::ostringstream outs;
	outs.fill(' ');
	if (_width != 0) outs.width(_width);
	if (_prec == 0) _prec = bestPrecision(F, _width);
	if (_prec == -1 && _width > 7)
	{
		outs.precision(_width - 7);
		outs.setf(std::ios::scientific);
	}
	else
		outs.precision(_prec < 0 ? 6 : _prec); // libstdc++ treats a negative precision as 6
	outs << F;
	return outs.str();
}

inline std::string integerToString(int I) { return std::to_string(I); }

inline int textToInteger(const char *str, int _errno = 0, std::string errmsg = "")
{
	int retval;
	if (str == NULL || sscanf(str, "%d", &retval) != 1) REPORT_ERROR(errmsg);
	return retval;
}
inline int textToInteger(const std::string &str, int _errno = 0, std::string errmsg = "") { return textToInteger(str.c_str(), _errno, errmsg); }

inline float textToFloat(const char *str, int _errno = 0, std::string errmsg = "")
{
	float retval;
	if (str == NULL || sscanf(str, "%f", &retval) != 1) REPORT_ERROR(errmsg);
	return retval;
}
inline float textToFloat(const std::string &str, int _errno = 0, std::string errmsg = "") { return textToFloat(str.c_str(), _errno, errmsg); }

inline bool textToBool(const char *str, int _errno = 0, std::string errmsg = "")
{
	if (str == NULL) REPORT_ERROR(errmsg);
	if (strcasecmp(str, "true") == 0 || strcasecmp(str, "yes") == 0) return true;
	if (strcasecmp(str, "false") == 0 || strcasecmp(str, "no") == 0) return false;
	REPORT_ERROR(errmsg);
}

typedef std::string FileName;

// --- constants from pipeline_jobs.h (pasted by extract.py) ----------------------------------
// @@PIPELINE_JOBS_H_DEFINES@@

// --- JobOption ------------------------------------------------------------------------------
class JobOption
{
public:
	std::string label, label_gui, variable, value, default_value, helptext;
	int joboption_type;
	float min_value, max_value, step_value;
	std::string node_type, pattern, directory;
	int node_type_depth;
	std::vector<std::string> radio_options;

	JobOption() { clear(); }
	// Any constructor
	JobOption(std::string _label, std::string _default_value, std::string _helptext)
	{
		clear(); initialise(_label, _default_value, _helptext); joboption_type = JOBOPTION_ANY;
	}
	// FileName constructor
	JobOption(std::string _label, std::string _default_value, std::string _pattern, std::string _directory, std::string _helptext)
	{
		clear(); initialise(_label, _default_value, _helptext); joboption_type = JOBOPTION_FILENAME;
		pattern = _pattern; directory = _directory;
	}
	// InputNode constructor
	JobOption(std::string _label, std::string _nodetype, int _node_type_depth, std::string _default_value, std::string _pattern, std::string _helptext)
	{
		clear(); initialise(_label, _default_value, _helptext); joboption_type = JOBOPTION_INPUTNODE;
		pattern = _pattern; node_type_depth = _node_type_depth; node_type = _nodetype;
	}
	// Radio constructor
	JobOption(std::string _label, std::vector<std::string> _radio_options, int ioption, std::string _helptext)
	{
		clear(); radio_options = _radio_options;
		initialise(_label, radio_options[ioption], _helptext); joboption_type = JOBOPTION_RADIO;
	}
	// Boolean constructor
	JobOption(std::string _label, bool _boolvalue, std::string _helptext)
	{
		clear(); initialise(_label, _boolvalue ? "Yes" : "No", _helptext); joboption_type = JOBOPTION_BOOLEAN;
	}
	// Slider constructor
	JobOption(std::string _label, float _default_value, float _min_value, float _max_value, float _step_value, std::string _helptext)
	{
		clear(); initialise(_label, floatToString(_default_value), _helptext); joboption_type = JOBOPTION_SLIDER;
		min_value = _min_value; max_value = _max_value; step_value = _step_value;
	}
	void clear()
	{
		label = value = default_value = helptext = label_gui = pattern = directory = "undefined";
		joboption_type = JOBOPTION_UNDEFINED;
		radio_options = job_undefined_options;
		node_type = "";
		node_type_depth = 1;
		min_value = max_value = step_value = 0.;
	}
	void initialise(std::string _label, std::string _default_value, std::string _helptext)
	{
		label = label_gui = _label;
		value = default_value = _default_value;
		helptext = _helptext;
	}
	std::string getString() { return value; }
	void setString(std::string s) { value = s; }
	bool getBoolean() { return value == "Yes"; }
};

class Node
{
public:
	std::string name, type;
	int type_depth;
	Node(std::string _name, std::string _type, int _type_depth = 1) : name(_name), type(_type), type_depth(_type_depth) {}
};

// --- RelionJob ------------------------------------------------------------------------------
class RelionJob
{
public:
	std::string outputName, alias, hidden_name;
	int type;
	FileName label;
	bool is_continue, is_tomo;
	std::map<std::string, JobOption> joboptions;

	RelionJob() { type = -1; is_continue = is_tomo = false; }
	void setTomo(bool _is_tomo) { is_tomo = _is_tomo; }
	bool read(std::string fn, bool &_is_continue, bool do_initialise = false) { return false; }

	void initialise(int job_type);
	void addTomoInputOptions(bool has_tomograms, bool has_particles, bool has_trajectories, bool has_manifolds = false);
	void initialiseImportJob();
	void initialiseMotioncorrJob();
	void initialiseCtffindJob();
	void initialiseManualpickJob();
	void initialiseAutopickJob();
	void initialiseExtractJob();
	void initialiseSelectJob();
	void initialiseClass2DJob();
	void initialiseInimodelJob();
	void initialiseClass3DJob();
	void initialiseAutorefineJob();
	void initialiseMultiBodyJob();
	void initialiseMaskcreateJob();
	void initialiseJoinstarJob();
	void initialiseSubtractJob();
	void initialisePostprocessJob();
	void initialiseLocalresJob();
	void initialiseDynaMightJob();
	void initialiseModelAngeloJob();
	void initialiseMotionrefineJob();
	void initialiseCtfrefineJob();
	void initialiseExternalJob();
	void initialiseTomoImportJob();
	void initialiseTomoAlignTiltSeriesJob();
	void initialiseTomoReconstructTomogramsJob();
	void initialiseTomoDenoiseTomogramsJob();
	void initialiseTomoPickTomogramsJob();
	void initialiseTomoExcludeTiltImagesJob();
	void initialiseTomoSubtomoJob();
	void initialiseTomoCtfRefineJob();
	void initialiseTomoAlignJob();
	void initialiseTomoReconPartJob();
};

// --- FLTK stand-ins -------------------------------------------------------------------------
#define MENUHEIGHT 30
#define TABHEIGHT 25
#define GUIWIDTH 800
#define GUIEXTRA 0
#define GUIHEIGHT_OLD 420 + GUIEXTRA
#define XCOL0 200
#define WCOL0 200
#define XCOL1 ((XCOL0) + 10)
#define XCOL2 ((XCOL0) + 280)
#define XCOL3 ((XCOL0) + 480)
#define XCOL4 ((XCOL0) + 495)
#define XCOL5 ((XCOL0) + 555)
#define STEPY 20
#define COLUMN_SEPARATION 3
#define WCOL1 ((XCOL2) - (XCOL1) - (COLUMN_SEPARATION))
#define WCOL2 ((XCOL3) - (XCOL2) - (COLUMN_SEPARATION))
#define WCOL3 ((XCOL4) - (XCOL3) - (COLUMN_SEPARATION))
#define WCOL4 ((XCOL5) - (XCOL4) - (COLUMN_SEPARATION))
#define GUI_BACKGROUND_COLOR 0
#define GUI_BACKGROUND_COLOR2 0

class Fl_Group;
struct Recorder;
Recorder &rec();

class Fl_Group
{
public:
	static Fl_Group *current_;
	Fl_Group *parent;
	std::string label_;
	bool is_tabs;
	int id;
	Fl_Group(int, int, int, int, const char *l = 0);
	virtual ~Fl_Group() {}
	void begin() { current_ = this; }
	void end() { current_ = parent; }
	void label(const char *l) { label_ = l ? l : ""; }
	void color(int) {}
	void selection_color(int) {}
};

class Fl_Tabs : public Fl_Group
{
public:
	Fl_Tabs(int x, int y, int w, int h, const char *l = 0) : Fl_Group(x, y, w, h, l) { is_tabs = true; }
};

struct Placement
{
	std::vector<std::string> keys;
	std::string row_label; // only for place2/place3 rows
	int y;
	Fl_Group *container;
};

struct GroupInfo
{
	Fl_Group *g;
	std::string controller; // boolean option that (de)activates this group
	bool actually_activate; // RELION's flag: true means the group is active when the controller is "No"
};

struct Recorder
{
	std::vector<Fl_Group *> all_groups;
	std::vector<Placement> placements;
	std::map<Fl_Group *, GroupInfo> controllers;
	void reset()
	{
		all_groups.clear();
		placements.clear();
		controllers.clear();
		Fl_Group::current_ = NULL;
	}
};

inline Fl_Group::Fl_Group(int, int, int, int, const char *l)
{
	parent = current_;
	label_ = l ? l : "";
	is_tabs = false;
	id = (int)rec().all_groups.size();
	rec().all_groups.push_back(this);
}

class GuiEntry
{
public:
	int deactivate_option;
	int forced_state; // -1: none, 0: forced active, 1: forced inactive (GuiEntry::deactivate calls)
	GuiEntry() : deactivate_option(TOGGLE_LEAVE_ACTIVE), forced_state(-1) {}
	void deactivate(bool do_deactivate = true) { forced_state = do_deactivate ? 1 : 0; }
	void cb_menu_i() {}
};

// --- JobWindow ------------------------------------------------------------------------------
class JobWindow
{
public:
	bool is_continue, is_tomo;
	std::map<std::string, GuiEntry> guientries;
	RelionJob myjob;
	int x, y, w, h;
	int start_y, current_y;
	Fl_Tabs *tabs;
	Fl_Group *tab1, *tab2, *tab3, *tab4, *tab5, *tab6, *tab7, *runtab;
	Fl_Group *group0, *group1, *group2, *group3, *group4, *group5, *group6, *group7, *group8, *group9, *queue_group;

	JobWindow(int _x = WCOL0, int _y = 2, int _w = GUIWIDTH - WCOL0 - 10, int _h = GUIHEIGHT_OLD - 65, const char *title = "")
	{
		tabs = NULL;
		tab1 = tab2 = tab3 = tab4 = tab5 = tab6 = tab7 = runtab = NULL;
		group0 = group1 = group2 = group3 = group4 = group5 = group6 = group7 = group8 = group9 = queue_group = NULL;
		current_y = start_y = 0;
		is_continue = is_tomo = false;
		x = _x; y = _y; w = _w; h = _h;
	}

	void setupTabs(int nr_tabs);
	void setupRunTab();
	void resetHeight() { current_y = start_y; }
	void updateMyGui() {}
	void initialise(int my_job_type, bool _is_tomo = false);

	void record(std::vector<std::string> keys, std::string row_label, int deactivate_option)
	{
		for (size_t i = 0; i < keys.size(); i++)
			guientries[keys[i]].deactivate_option = deactivate_option;
		rec().placements.push_back(Placement{keys, row_label, current_y, Fl_Group::current_});
		current_y += STEPY + 2;
	}
	void place(std::string key, int deactivate_option = TOGGLE_LEAVE_ACTIVE, Fl_Group *deactivate_this_group = NULL, bool actually_activate = false)
	{
		record({key}, "", deactivate_option);
		if (deactivate_this_group != NULL)
			rec().controllers[deactivate_this_group] = GroupInfo{deactivate_this_group, key, actually_activate};
	}
	void place2(std::string key1, std::string key2, std::string label, int deactivate_option = TOGGLE_LEAVE_ACTIVE)
	{
		record({key1, key2}, label, deactivate_option);
	}
	void place3(std::string key1, std::string key2, std::string key3, std::string label, int deactivate_option = TOGGLE_LEAVE_ACTIVE)
	{
		record({key1, key2, key3}, label, deactivate_option);
	}

	void initialiseImportWindow();
	void initialiseMotioncorrWindow();
	void initialiseCtffindWindow();
	void initialiseManualpickWindow();
	void initialiseAutopickWindow();
	void initialiseExtractWindow();
	void initialiseSelectWindow();
	void initialiseClass2DWindow();
	void initialiseInimodelWindow();
	void initialiseClass3DWindow();
	void initialiseAutorefineWindow();
	void initialiseMultiBodyWindow();
	void initialiseMaskcreateWindow();
	void initialiseJoinstarWindow();
	void initialiseSubtractWindow();
	void initialisePostprocessWindow();
	void initialiseLocresWindow();
	void initialiseMotionrefineWindow();
	void initialiseCtfrefineWindow();
	void initialiseDynaMightWindow();
	void initialiseModelAngeloWindow();
	void initialiseExternalWindow();
	void placeTomoInput(bool has_tomograms, bool has_particles, bool has_trajectories, bool has_manifolds);
	void initialiseTomoImportWindow();
	void initialiseTomoAlignTiltseriesWindow();
	void initialiseTomoReconstructTomogramsWindow();
	void initialiseTomoDenoiseTomogramsWindow();
	void initialiseTomoPickTomogramsWindow();
	void initialiseTomoSubtomoWindow();
	void initialiseTomoCtfRefineWindow();
	void initialiseTomoAlignWindow();
	void initialiseTomoReconParWindow();
	void initialiseTomoExcludeTiltImagesWindow();
};

class Fl_Hold_Browser
{
public:
	std::vector<std::string> items;
	Fl_Hold_Browser(int, int, int, int) {}
	void add(const char *s) { items.push_back(s); }
	void textsize(int) {}
};
