// Utilities for parsing/converting protobufs and initializing SCAMP arguments

#pragma once

#include "../src/common.h"
#include "scamp.pb.h"

SCAMP::SCAMPPrecisionType ConvertPrecisionType(
    const SCAMPProto::SCAMPPrecisionType &t);

SCAMP::SCAMPProfileType ConvertProfileType(
    const SCAMPProto::SCAMPProfileType &t);

SCAMPProto::Profile ConvertProfile(const SCAMP::Profile &p);

int64_t GetProfileSize(const SCAMP::Profile &p);

SCAMPProto::SCAMPArgs ConvertArgsToReply(const SCAMP::SCAMPArgs &args);

void ConvertProtoArgsToSCAMPArgs(const SCAMPProto::SCAMPArgs &proto_args,
                                 SCAMP::SCAMPArgs *args);
