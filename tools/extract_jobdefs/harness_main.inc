// Copyright (C) 2026 Adrián González-López
// SPDX-License-Identifier: GPL-2.0-or-later
// Driver for the job-definition harness. Included at the end of the generated source, after the
// verbatim RELION functions. Prints one JSON document to stdout.

Fl_Group *Fl_Group::current_ = NULL;
Recorder &rec()
{
	static Recorder r;
	return r;
}

static std::string jstr(const std::string &s)
{
	std::string o = "\"";
	for (unsigned char c : s)
	{
		switch (c)
		{
		case '"': o += "\\\""; break;
		case '\\': o += "\\\\"; break;
		case '\n': o += "\\n"; break;
		case '\t': o += "\\t"; break;
		case '\r': o += "\\r"; break;
		default:
			if (c < 0x20)
			{
				char buf[8];
				snprintf(buf, sizeof(buf), "\\u%04x", c);
				o += buf;
			}
			else
				o += (char)c;
		}
	}
	return o + "\"";
}

static std::string jnum(float f)
{
	std::ostringstream o;
	o.precision(9);
	o << f;
	return o.str();
}

static const char *kind_name(int t)
{
	switch (t)
	{
	case JOBOPTION_ANY: return "any";
	case JOBOPTION_FILENAME: return "filename";
	case JOBOPTION_INPUTNODE: return "inputnode";
	case JOBOPTION_RADIO: return "radio";
	case JOBOPTION_BOOLEAN: return "boolean";
	case JOBOPTION_SLIDER: return "slider";
	case JOBOPTION_ONLYTEXT: return "onlytext";
	default: return "undefined";
	}
}

static const char *toggle_name(int t)
{
	switch (t)
	{
	case TOGGLE_DEACTIVATE: return "deactivate";
	case TOGGLE_REACTIVATE: return "reactivate";
	case TOGGLE_ALWAYS_DEACTIVATE: return "always_deactivate";
	default: return "leave_active";
	}
}

// Nearest enclosing group that is a tab page (its parent is the Fl_Tabs)
static Fl_Group *tab_of(Fl_Group *g)
{
	while (g != NULL && !(g->parent != NULL && g->parent->is_tabs))
		g = g->parent;
	return g;
}

// Nearest enclosing ordinary group (not a tab page, not the Fl_Tabs), or NULL
static Fl_Group *group_of(Fl_Group *g)
{
	if (g == NULL || g->is_tabs || (g->parent != NULL && g->parent->is_tabs)) return NULL;
	return g;
}

static std::string dump_window(JobWindow *w)
{
	Recorder &r = rec();
	std::ostringstream o;
	RelionJob &job = w->myjob;
	o << "{\"proc_id\":" << job.type << ",\"is_tomo\":" << (job.is_tomo ? "true" : "false");

	// options
	o << ",\"options\":{";
	bool first = true;
	for (auto &kv : job.joboptions)
	{
		JobOption &jo = kv.second;
		if (!first) o << ",";
		first = false;
		o << jstr(kv.first) << ":{\"kind\":" << jstr(kind_name(jo.joboption_type))
		  << ",\"label\":" << jstr(jo.label) << ",\"default\":" << jstr(jo.default_value)
		  << ",\"help\":" << jstr(jo.helptext);
		if (jo.joboption_type == JOBOPTION_SLIDER)
			o << ",\"min\":" << jnum(jo.min_value) << ",\"max\":" << jnum(jo.max_value) << ",\"step\":" << jnum(jo.step_value);
		if (jo.joboption_type == JOBOPTION_RADIO)
		{
			o << ",\"choices\":[";
			for (size_t i = 0; i < jo.radio_options.size(); i++)
				o << (i ? "," : "") << jstr(jo.radio_options[i]);
			o << "]";
		}
		if (jo.joboption_type == JOBOPTION_FILENAME)
			o << ",\"pattern\":" << jstr(jo.pattern) << ",\"directory\":" << jstr(jo.directory);
		if (jo.joboption_type == JOBOPTION_INPUTNODE)
			o << ",\"pattern\":" << jstr(jo.pattern) << ",\"node_type\":" << jstr(jo.node_type)
			  << ",\"node_type_depth\":" << jo.node_type_depth;
		auto ge = w->guientries.find(kv.first);
		if (ge != w->guientries.end())
		{
			o << ",\"toggle\":" << jstr(toggle_name(ge->second.deactivate_option));
			if (ge->second.forced_state >= 0)
				o << ",\"forced_inactive\":" << (ge->second.forced_state == 1 ? "true" : "false");
		}
		o << "}";
	}
	o << "}";

	// tabs, in creation order
	std::vector<Fl_Group *> tabs;
	for (Fl_Group *g : r.all_groups)
		if (g->parent != NULL && g->parent->is_tabs)
			tabs.push_back(g);
	o << ",\"tabs\":[";
	for (size_t t = 0; t < tabs.size(); t++)
	{
		o << (t ? "," : "") << "{\"label\":" << jstr(tabs[t]->label_) << ",\"rows\":[";
		bool firstrow = true;
		for (Placement &p : r.placements)
		{
			if (tab_of(p.container) != tabs[t]) continue;
			o << (firstrow ? "" : ",") << "{\"keys\":[";
			firstrow = false;
			for (size_t k = 0; k < p.keys.size(); k++)
				o << (k ? "," : "") << jstr(p.keys[k]);
			o << "],\"y\":" << p.y;
			if (!p.row_label.empty()) o << ",\"label\":" << jstr(p.row_label);
			Fl_Group *g = group_of(p.container);
			if (g != NULL) o << ",\"group\":" << g->id;
			o << "}";
		}
		o << "]}";
	}
	o << "]";

	// groups that hold rows or have a controller
	o << ",\"groups\":{";
	first = true;
	for (Fl_Group *g : r.all_groups)
	{
		if (group_of(g) == NULL || tab_of(g) == NULL) continue;
		if (!first) o << ",";
		first = false;
		o << jstr(std::to_string(g->id)) << ":{\"parent\":";
		Fl_Group *pg = group_of(g->parent);
		if (pg != NULL) o << pg->id; else o << "null";
		auto c = r.controllers.find(g);
		if (c != r.controllers.end())
			o << ",\"controller\":" << jstr(c->second.controller)
			  << ",\"active_when\":" << jstr(c->second.actually_activate ? "No" : "Yes");
		else
			o << ",\"controller\":null";
		o << "}";
	}
	o << "}";

	// placements referring to undefined options
	o << ",\"missing\":[";
	first = true;
	std::set<std::string> seen;
	for (Placement &p : r.placements)
		for (auto &k : p.keys)
			if (job.joboptions.find(k) == job.joboptions.end() && seen.insert(k).second)
			{
				o << (first ? "" : ",") << jstr(k);
				first = false;
			}
	o << "]}";
	return o.str();
}

static std::string run_window(int type, bool tomo)
{
	rec().reset();
	JobWindow *w = new JobWindow();
	w->initialise(type, tomo);
	std::string out = dump_window(w);
	delete w;
	rec().reset();
	return out;
}

struct MenuEntry
{
	std::string label;
	int type;
	bool tomo;
};
static std::vector<MenuEntry> g_menu;

static void harness_init(JobWindow *w, Fl_Hold_Browser *b, int type, bool tomo = false)
{
	g_menu.push_back(MenuEntry{b->items.empty() ? "" : b->items.back(), type, tomo});
	(void)w;
}

void build_menu(bool _do_tomo);

int main()
{
	try
	{
		std::ostringstream o;
		o << "{\"proc_types\":[";
		bool first = true;
		for (auto &kv : proc_type2labelnew)
		{
			o << (first ? "" : ",") << "{\"proc_id\":" << kv.first << ",\"label\":" << jstr(kv.second)
			  << ",\"dirname\":" << jstr(proc_type2dirname.at(kv.first)) << "}";
			first = false;
		}
		o << "],\"variants\":[";
		first = true;
		for (auto &kv : proc_type2labelnew)
			for (int tomo = 0; tomo < 2; tomo++)
			{
				o << (first ? "" : ",") << run_window(kv.first, tomo == 1);
				first = false;
			}
		o << "],\"menus\":{";
		for (int tomo = 0; tomo < 2; tomo++)
		{
			g_menu.clear();
			build_menu(tomo == 1);
			o << (tomo ? "," : "") << jstr(tomo ? "tomo" : "spa") << ":[";
			for (size_t i = 0; i < g_menu.size(); i++)
				o << (i ? "," : "") << "{\"menu_label\":" << jstr(g_menu[i].label) << ",\"proc_id\":" << g_menu[i].type
				  << ",\"is_tomo\":" << (g_menu[i].tomo ? "true" : "false") << "}";
			o << "]";
		}
		o << "}}";
		std::cout << o.str() << std::endl;
	}
	catch (RelionError &e)
	{
		std::cerr << "RelionError: " << e.msg << std::endl;
		return 1;
	}
	return 0;
}
