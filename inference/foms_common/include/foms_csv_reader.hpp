#ifndef FOMS_CSVREADER_HPP
#define FOMS_CSVREADER_HPP

#include <vector>
#include <map>
#include <string>
#include <iostream>
#include <sys/time.h>
#include <stdint.h>
/**
 * @file
 * @brief Defines class for csv reader.
 */

namespace foms {

/// @brief  class for csv reader
class CsvReader {

  public:
    /// @brief default constructor
    CsvReader() = delete;
    /// @brief Contructor with a filename
    /// @param filename path of the csv file
    /// @param sep character separator
    explicit CsvReader(std::string_view filename, char sep = ',');
    /// @brief destructor
    virtual ~CsvReader() = default;

    /// @brief checks if the file is loaded
    /// @return true if the file is loaded , false otherwise
    bool is_loaded() const { return m_loaded; }

    /// @brief get the nomber of lines in the csv
    /// @return the number of lines in the csv
    size_t getNumberofLines() const { return m_values.size(); }

    /// @brief get a value by index
    /// @param index index of the value in the line
    /// @param lineNumber line number
    /// @return the string value

    std::string getValueByIndex(unsigned int index, unsigned int lineNumber) const;

    /// @brief get a value by name
    /// @param valueName name of the value in the line
    /// @param lineNumber line number
    /// @return the string value
    std::string getValueByName(const std::string& valueName, int lineNumber) const;

  private:
    /// @brief csv file
    std::string m_file;
    /// @brief state of the reader (loaded/not loaded)
    bool m_loaded = false;
    /// @brief csv separator
    char m_separator{','};

    /// @brief map for the header
    std::map< std::string, int > m_map_header;
    /// @brief list of line values
    std::vector< std::vector< std::string > > m_values;
};

} // namespace foms

#endif // FOMS_CSVREADER_HPP