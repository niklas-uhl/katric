/*
 * Copyright (c) 2020-2023 Tim Niklas Uhl
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#pragma once

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <CLI/CLI.hpp>
#include <CLI/Validators.hpp>

#include "cetric/config.h"
#include "magic_enum/magic_enum.hpp"

/// Adds an option for an enum. Its values are spelled as cetric::cli_name() (case-insensitive, `_` for `-` works as
/// well), `aliases` are accepted in addition. The help text and option_values() show the value by name.
template <typename Enum>
CLI::Option*
add_enum_option(CLI::App& app, std::string const& name, Enum& variable, std::map<std::string, Enum> aliases = {}) {
    std::map<std::string, Enum> values;
    for (Enum value: magic_enum::enum_values<Enum>()) {
        values.emplace(cetric::cli_name(value), value);
    }
    values.merge(aliases);
    std::vector<std::string> names;
    for (auto const& [name, value]: values) {
        names.push_back(name);
    }
    auto underscores_as_hyphens = [](std::string str) {
        std::replace(str.begin(), str.end(), '_', '-');
        return str;
    };
    // IsMember normalizes the name and gives the help text and error message, then the (unchecked) Transformer
    // converts it. Transforms run in reverse order of addition.
    return app.add_option(name, variable)
        ->transform(CLI::Transformer(values, CLI::ignore_case, underscores_as_hyphens).description(""))
        ->transform(CLI::IsMember(names, CLI::ignore_case, underscores_as_hyphens))
        ->default_function([&variable] { return cetric::cli_name(variable); })
        ->capture_default_str();
}

/// Name and parsed value of every option of `app`, e.g. to record the configuration of a run. Call it after parsing
/// (and after adjusting parsed values, which are read from the bound variables).
inline std::vector<std::pair<std::string, std::string>> option_values(CLI::App& app) {
    std::vector<std::pair<std::string, std::string>> values;
    for (CLI::Option* option: app.get_options()) {
        auto name = option->get_single_name();
        if (name == "help") {
            continue;
        }
        if (option->get_expected_min() == 0) { // flags capture no default; they all default to false
            values.emplace_back(name, option->count() > 0 ? option->as<std::string>() : "false");
        } else {
            option->capture_default_str(); // formats the bound variable, i.e. the parsed value
            values.emplace_back(name, option->get_default_str());
        }
    }
    return values;
}

#ifndef CLI11_PARSE_MPI
    #define CLI11_PARSE_MPI(app, argc, argv, rank, size) \
        int retval = -1;                                 \
        try {                                            \
            (app).parse((argc), (argv));                 \
        } catch (const CLI::ParseError& e) {             \
            if (rank == 0) {                             \
                retval = (app).exit(e);                  \
            } else {                                     \
                retval = 1;                              \
            }                                            \
        }                                                \
        if (retval > -1) {                               \
            MPI_Finalize();                              \
            exit(retval);                                \
        }
#endif

