// -----------------------------------------------------------------------------------------
// QSVEnc/NVEnc/VCEEnc by rigaya
// -----------------------------------------------------------------------------------------
//
// The MIT License
//
// Copyright (c) 2014-2016 rigaya
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//
// ------------------------------------------------------------------------------------------

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <string>
#include "rgy_util.h"      // strsprintf (brought in transitively via header too, make explicit to avoid ordering quirks)
#include "rgy_d2v_parser.h"

RGYD2VParser::RGYD2VParser()
    : m_frames(), m_path(), m_progressive(0), m_rff(0), m_interlaced(0) {
}

RGYD2VParser::~RGYD2VParser() {}

// Line-at-a-time reader that works with _tfopen'd FILE* on both narrow and
// wide-char Windows builds. D2V files are ASCII so fgets into a char buffer
// is fine regardless of tstring's character width.
static bool readLineAscii(FILE *fp, std::string &out) {
    char buf[4096];
    out.clear();
    while (fgets(buf, sizeof(buf), fp)) {
        const size_t len = strlen(buf);
        if (len == 0) return !out.empty();
        // Strip trailing CR/LF.
        size_t end = len;
        while (end > 0 && (buf[end - 1] == '\r' || buf[end - 1] == '\n')) end--;
        out.append(buf, end);
        // If fgets did not hit a newline, the line is longer than the buffer;
        // loop to keep appending.
        if (end == len && buf[len - 1] != '\n') {
            continue;
        }
        return true;
    }
    return !out.empty();
}

bool RGYD2VParser::load(const tstring &path) {
    m_frames.clear();
    m_progressive = 0;
    m_rff = 0;
    m_interlaced = 0;
    m_path = path;

    FILE *fp = _tfopen(path.c_str(), _T("r"));
    if (!fp) {
        return false;
    }

    std::string line;

    // --- Header ---
    // Line 1 must start with "DGIndexProjectFile".
    if (!readLineAscii(fp, line) || line.compare(0, 18, "DGIndexProjectFile") != 0) {
        fclose(fp);
        return false;
    }
    // Line 2: file count.
    if (!readLineAscii(fp, line)) { fclose(fp); return false; }
    const int fileCount = std::atoi(line.c_str());
    if (fileCount < 0 || fileCount > 10000) { fclose(fp); return false; }
    // Next fileCount lines: source file paths.
    for (int i = 0; i < fileCount; i++) {
        if (!readLineAscii(fp, line)) { fclose(fp); return false; }
    }
    // Blank line terminator.
    if (!readLineAscii(fp, line)) { fclose(fp); return false; }
    // Tolerate but don't require strict blank — some tools write whitespace.

    // --- Settings section ---
    // Lines with "Key=Value" until blank. Consume unconditionally.
    while (readLineAscii(fp, line)) {
        // Strip trailing whitespace.
        while (!line.empty() && (line.back() == ' ' || line.back() == '\t')) line.pop_back();
        if (line.empty()) break;
    }

    // --- Data section ---
    // Each line is one GOP: "info matrix file position skip vob cell flag0 flag1 ..."
    // First 7 tokens are headers (info .. cell); remaining are per-picture
    // flag bytes in display order, space-separated hex. "ff" is EOS.
    // Stop on a "FINISHED" summary line or EOF.
    while (readLineAscii(fp, line)) {
        if (line.empty()) continue;
        if (line[0] == '#') continue;
        if (line.compare(0, 8, "FINISHED") == 0) break;

        // Tokenize by whitespace.
        std::vector<std::string> tokens;
        size_t i = 0;
        while (i < line.size()) {
            while (i < line.size() && std::isspace((unsigned char)line[i])) i++;
            if (i >= line.size()) break;
            const size_t start = i;
            while (i < line.size() && !std::isspace((unsigned char)line[i])) i++;
            tokens.emplace_back(line.substr(start, i - start));
        }
        if (tokens.size() <= 7) continue;

        for (size_t t = 7; t < tokens.size(); t++) {
            const std::string &s = tokens[t];
            if (s == "ff" || s == "FF") continue;  // end-of-stream marker
            // Parse hex byte; reject anything out of range.
            char *endp = nullptr;
            const unsigned long v = std::strtoul(s.c_str(), &endp, 16);
            if (endp == s.c_str() || v > 0xff) continue;
            const uint8_t b = (uint8_t)v;

            D2VFrameInfo info;
            info.progressive = (uint8_t)((b >> 6) & 1);
            info.pictureType = (uint8_t)((b >> 4) & 3);
            info.tff         = (uint8_t)((b >> 1) & 1);
            info.rff         = (uint8_t)( b       & 1);
            m_frames.push_back(info);

            if (info.progressive) m_progressive++;
            else                  m_interlaced++;
            if (info.rff)         m_rff++;
        }
    }

    fclose(fp);
    return !m_frames.empty();
}

tstring RGYD2VParser::stats() const {
    if (m_frames.empty()) {
        return _T("d2v: no frames loaded");
    }
    // NOTE: use %lld with explicit (long long) cast instead of %zu. MSVC's
    // wide-char runtime has historically been inconsistent about the z length
    // specifier; some binaries treat it as an unrecognized specifier and
    // crash or emit garbage. %lld with the explicit cast is portable.
    const long long n = (long long)m_frames.size();
    const double pctProg = 100.0 * (double)m_progressive / (double)m_frames.size();
    const double pctRff  = 100.0 * (double)m_rff         / (double)m_frames.size();
    const double pctInt  = 100.0 * (double)m_interlaced  / (double)m_frames.size();
    return strsprintf(
        _T("d2v: %lld frames (progressive %.2f%%, RFF %.2f%%, interlaced %.2f%%)"),
        n, pctProg, pctRff, pctInt);
}
