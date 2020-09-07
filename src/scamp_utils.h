#pragma once

#include <fstream>
#include <string>
#include <vector>
#include "common.h"

void write_matrix(const std::string &mp, bool output_pearson,
                  const std::vector<std::vector<double>> &matrix, int window);
std::vector<std::vector<double>> reduce_all_neighbors(SCAMP::ProfileData *data,
                                                      int height, int width,
                                                      int output_height,
                                                      int output_width);

std::ifstream &read_value(std::ifstream &s, double &d, int count);

void readFile(const std::string &filename, std::vector<double> &v);

std::vector<int> ParseIntList(const std::string &s);

SCAMP::SCAMPPrecisionType GetPrecisionType(bool ultrap, bool doublep,
                                           bool mixedp, bool singlep);

SCAMP::SCAMPProfileType ParseProfileType(const std::string &s);

double ConvertToEuclidean(double val, int window);
double CleanupPearson(double val);

bool WriteProfileToFile(const std::string &mp, const std::string &mpi,
                        SCAMP::Profile &p, bool output_pearson, int window,
                        int matrix_width, int matrix_height);

bool InitProfileMemory(SCAMP::SCAMPArgs *args);
