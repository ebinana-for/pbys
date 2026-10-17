#include "foms_csv_reader.hpp"
#include "foms_logStream.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <map>

foms::CsvReader::CsvReader(std::string_view filename, char sep)
    : m_file{filename},
      m_separator{sep}

{
    std::ifstream ifs(m_file);
    if (!ifs.is_open()) {
        foms::g_logger.writeLog("Unable to open file " + m_file, foms::errorLevel::ERROR);
        return;
    }

    //////////////read header///////////////////////////////////
    std::string header_line;
    getline(ifs, header_line);

    if (header_line.empty()) {
        foms::g_logger.writeLog("Unable to read header " + m_file, foms::errorLevel::ERROR);
        return;
    }

    std::stringstream s_header(header_line);
    std::string word{};
    int index = 0;
    while (getline(s_header, word, ',')) {

        m_map_header[word] = index++;
    }

    if (m_map_header.empty()) {
        foms::g_logger.writeLog("Unable to read header " + m_file, foms::errorLevel::ERROR);
        return;
    }

    //////////////read each line///////////////////////////////////
    std::string raw_line;

    while (!ifs.eof()) {

        std::vector< std::string > line_values;
        line_values.reserve(m_map_header.size());
        raw_line.clear();
        getline(ifs, raw_line);
        std::stringstream s_line(raw_line);

        std::string word_line{};
        while (getline(s_line, word_line, m_separator)) {
            line_values.push_back(word_line);
        }

        if (!line_values.empty() && line_values.size() != m_map_header.size()) {

            foms::g_logger.writeLog("Error number of values is no equal to the number of fields in header ",
                                    foms::errorLevel::ERROR);

            return;
        }
        m_values.push_back(line_values);
    }

    m_loaded = true;
}

std::string foms::CsvReader::getValueByIndex(unsigned int index, unsigned int lineNumber) const {

    std::string result;
    if (lineNumber >= m_values.size() || index > m_values[lineNumber].size()) {

        foms::g_logger.writeLog("Error bad line number or index ", foms::errorLevel::ERROR);

        return result;
    }

    result = m_values[lineNumber][index];

    return result;
}

std::string foms::CsvReader::getValueByName(const std::string& valueName, int lineNumber) const {
    std::string result;

    if (auto search = m_map_header.find(valueName); search != m_map_header.end()) {
        return getValueByIndex(search->second, lineNumber);
    }
    foms::g_logger.writeLog("Name not found in header ", foms::errorLevel::ERROR);
    return result;
}
