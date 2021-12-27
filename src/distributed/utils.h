// Utilities for parsing/converting protobufs and initializing SCAMP arguments

#pragma once

#include "common/common.h"
#include "common/scamp_args.h"
#include "scamp.pb.h"

int64_t get_current_time();

SCAMP::SCAMPPrecisionType ConvertPrecisionType(
    const SCAMPProto::SCAMPPrecisionType &t);

SCAMP::SCAMPProfileType ConvertProfileType(
    const SCAMPProto::SCAMPProfileType &t);

SCAMPProto::Profile ConvertProfile(const SCAMP::Profile &p);

int64_t GetProfileSize(const SCAMPProto::Profile &p);

SCAMPProto::SCAMPArgs ConvertSCAMPArgsToProtoArgs(const SCAMP::SCAMPArgs &args);

void ConvertProtoArgsToSCAMPArgs(const SCAMPProto::SCAMPArgs &proto_args,
                                 SCAMP::SCAMPArgs *args);

bool ProfileAllocated(const SCAMPProto::Profile &p);
bool InitProfile(SCAMPProto::Profile *p, SCAMPProto::SCAMPProfileType type,
                 int64_t size);
void MergeProfile(const SCAMPProto::SCAMPArgs &tile_args,
                  SCAMPProto::SCAMPArgs *job_args, SCAMPProto::Profile *a_tile,
                  uint64_t col_pos, uint64_t width, SCAMPProto::Profile *b_tile,
                  uint64_t row_pos, uint64_t height);

SCAMP::SCAMPArgs get_default_args(uint64_t input_size);

bool validateArgs(const SCAMPProto::SCAMPArgs &args);
