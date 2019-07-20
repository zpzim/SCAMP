#include <grpcpp/grpcpp.h>
#include "scamp.grpc.pb.h"

#include "scamp_interface.h"
#include "utils.h"

SCAMP::SCAMPPrecisionType ConvertPrecisionType(
    const SCAMPProto::SCAMPPrecisionType &t) {
  switch (t) {
    case SCAMPProto::PRECISION_DOUBLE:
      return SCAMP::PRECISION_DOUBLE;
    case SCAMPProto::PRECISION_MIXED:
      return SCAMP::PRECISION_MIXED;
    case SCAMPProto::PRECISION_SINGLE:
      return SCAMP::PRECISION_SINGLE;
    default:
      return SCAMP::PRECISION_INVALID;
  }
}

SCAMP::SCAMPProfileType ConvertProfileType(
    const SCAMPProto::SCAMPProfileType &t) {
  switch (t) {
    case SCAMPProto::PROFILE_TYPE_1NN_INDEX:
      return SCAMP::PROFILE_TYPE_1NN_INDEX;
    case SCAMPProto::PROFILE_TYPE_1NN:
      return SCAMP::PROFILE_TYPE_1NN;
    case SCAMPProto::PROFILE_TYPE_SUM_THRESH:
      return SCAMP::PROFILE_TYPE_SUM_THRESH;
    default:
      return SCAMP::PROFILE_TYPE_INVALID;
  }
}

SCAMPProto::SCAMPPrecisionType ConvertPrecisionType(
    const SCAMP::SCAMPPrecisionType &t) {
  switch (t) {
    case SCAMP::PRECISION_DOUBLE:
      return SCAMPProto::PRECISION_DOUBLE;
    case SCAMP::PRECISION_MIXED:
      return SCAMPProto::PRECISION_MIXED;
    case SCAMP::PRECISION_SINGLE:
      return SCAMPProto::PRECISION_SINGLE;
    default:
      return SCAMPProto::PRECISION_INVALID;
  }
}

SCAMPProto::SCAMPProfileType ConvertProfileType(
    const SCAMP::SCAMPProfileType &t) {
  switch (t) {
    case SCAMP::PROFILE_TYPE_1NN_INDEX:
      return SCAMPProto::PROFILE_TYPE_1NN_INDEX;
    case SCAMP::PROFILE_TYPE_1NN:
      return SCAMPProto::PROFILE_TYPE_1NN;
    case SCAMP::PROFILE_TYPE_SUM_THRESH:
      return SCAMPProto::PROFILE_TYPE_SUM_THRESH;
    default:
      return SCAMPProto::PROFILE_TYPE_INVALID;
  }
}

SCAMPProto::Profile ConvertProfile(const SCAMP::Profile &p) {
  // std::cout << "size = " << p.data.size() << std::endl;

  SCAMPProto::Profile out;
  out.set_type(ConvertProfileType(p.type));

  // TODO FIX
  if (p.data.empty()) {
    return out;
  }

  switch (p.type) {
    case SCAMP::PROFILE_TYPE_1NN_INDEX:
      *out.mutable_data()->Add()->mutable_uint64_value()->mutable_value() = {
          p.data[0].uint64_value.begin(), p.data[0].uint64_value.end()};
      break;
    case SCAMP::PROFILE_TYPE_1NN:
      *out.mutable_data()->Add()->mutable_float_value()->mutable_value() = {
          p.data[0].float_value.begin(), p.data[0].float_value.end()};
      break;
    case SCAMP::PROFILE_TYPE_SUM_THRESH:
      *out.mutable_data()->Add()->mutable_double_value()->mutable_value() = {
          p.data[0].double_value.begin(), p.data[0].double_value.end()};
      break;
  }
  return out;
}

int64_t GetProfileSize(const SCAMP::Profile &p) {
  if (p.data.empty()) {
    return 0;
  }
  switch (p.type) {
    case SCAMP::PROFILE_TYPE_1NN_INDEX:
      return p.data[0].uint64_value.size();
    case SCAMP::PROFILE_TYPE_1NN:
      return p.data[0].float_value.size();
    case SCAMP::PROFILE_TYPE_SUM_THRESH:
      return p.data[0].double_value.size();
  }
  return 0;
}

SCAMPProto::SCAMPArgs ConvertArgsToReply(const SCAMP::SCAMPArgs &args) {
  SCAMPProto::SCAMPArgs reply;
  *reply.mutable_timeseries_a() = {args.timeseries_a.begin(),
                                   args.timeseries_a.end()};
  *reply.mutable_timeseries_b() = {args.timeseries_b.begin(),
                                   args.timeseries_b.end()};
  *reply.mutable_profile_a() = ConvertProfile(args.profile_a);
  *reply.mutable_profile_b() = ConvertProfile(args.profile_b);
  reply.set_has_b(args.has_b);
  reply.set_window(args.window);
  reply.set_max_tile_size(args.max_tile_size);
  reply.set_distributed_start_row(args.distributed_start_row);
  reply.set_distributed_start_col(args.distributed_start_col);
  reply.set_timeseries_size_a(GetProfileSize(args.profile_a));
  reply.set_timeseries_size_b(GetProfileSize(args.profile_b));
  reply.set_profile_type(ConvertProfileType(args.profile_type));
  reply.set_precision_type(ConvertPrecisionType(args.precision_type));
  reply.set_distance_threshold(args.distance_threshold);
  reply.set_computing_rows(args.computing_rows);
  reply.set_computing_columns(args.computing_columns);
  reply.set_keep_rows_separate(args.keep_rows_separate);
  reply.set_is_aligned(args.is_aligned);

  return reply;
}

SCAMP::Profile ConvertProfile(const SCAMPProto::Profile &profile) {
  SCAMP::Profile out;
  out.type = ConvertProfileType(profile.type());
  switch (profile.type()) {
    case SCAMPProto::PROFILE_TYPE_1NN_INDEX:
      out.data.emplace_back();
      out.data[0].uint64_value = {
          profile.data().Get(0).uint64_value().value().begin(),
          profile.data().Get(0).uint64_value().value().end()};
      break;
    case SCAMPProto::PROFILE_TYPE_1NN:
      out.data.emplace_back();
      out.data[0].float_value = {
          profile.data().Get(0).float_value().value().begin(),
          profile.data().Get(0).float_value().value().end()};
      break;
    case SCAMPProto::PROFILE_TYPE_SUM_THRESH:
      out.data.emplace_back();
      out.data[0].double_value = {
          profile.data().Get(0).double_value().value().begin(),
          profile.data().Get(0).double_value().value().end()};
      break;
  }
  return out;
}

void ConvertProtoArgsToSCAMPArgs(const SCAMPProto::SCAMPArgs &proto_args,
                                 SCAMP::SCAMPArgs *args) {
  std::vector<double> Ta_h, Tb_h;

  for (int i = 0; i < proto_args.timeseries_size_a(); i++) {
    Ta_h.push_back(proto_args.timeseries_a()[i]);
  }

  for (int i = 0; i < proto_args.timeseries_size_b(); i++) {
    Tb_h.push_back(proto_args.timeseries_b()[i]);
  }

  args->max_tile_size = proto_args.max_tile_size();
  args->distributed_start_row = proto_args.distributed_start_row();
  args->distributed_start_col = proto_args.distributed_start_col();
  args->distance_threshold = proto_args.distance_threshold();
  args->computing_columns = proto_args.computing_columns();
  args->computing_rows = proto_args.computing_rows();
  args->profile_a = std::move(ConvertProfile(proto_args.profile_a()));
  if (proto_args.keep_rows_separate()) {
    args->profile_b = std::move(ConvertProfile(proto_args.profile_b()));
  }
  args->profile_type = ConvertProfileType(proto_args.profile_type());
  args->precision_type = ConvertPrecisionType(proto_args.precision_type());
  args->keep_rows_separate = proto_args.keep_rows_separate();
  args->is_aligned = proto_args.is_aligned();
  args->window = proto_args.window();
  args->has_b = proto_args.has_b();
  args->timeseries_a = std::move(Ta_h);
  args->timeseries_b = std::move(Tb_h);
}
