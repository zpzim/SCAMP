#pragma once

#include <fstream>
#include <string>
#include <vector>
#include "common.h"

std::ifstream &read_value(std::ifstream &s, double &d, int count);

void readFile(const std::string &filename, std::vector<double> &v);

std::vector<int> ParseIntList(const std::string &s);

SCAMP::SCAMPPrecisionType GetPrecisionType(bool doublep, bool mixedp,
                                           bool singlep);

SCAMP::SCAMPProfileType ParseProfileType(const std::string &s);

double ConvertToEuclidean(double val, int window);

bool WriteProfileToFile(const std::string &mp, const std::string &mpi,
                        SCAMP::Profile p, bool output_pearson, int window);

bool InitProfileMemory(SCAMP::SCAMPArgs *args);
