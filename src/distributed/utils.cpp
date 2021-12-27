#include <grpcpp/grpcpp.h>
#include <random>

#include "common/profile.h"
#include "common/scamp_interface.h"
#include "common/scamp_utils.h"
#include "scamp.grpc.pb.h"
#include "utils.h"

int64_t get_current_time() {
  return std::chrono::duration_cast<std::chrono::seconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

// Converts between SCAMP:: and SCAMPProto:: precision types
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

// Converts between SCAMP:: and SCAMPProto:: profile types
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

// Converts between SCAMP:: and SCAMPProto:: precision types
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

// Converts between SCAMP:: and SCAMPProto:: profile types
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

// Converts between SCAMP:: and SCAMPProto:: profile formats
SCAMPProto::Profile ConvertProfile(const SCAMP::Profile &p) {
  SCAMPProto::Profile out;
  out.set_type(ConvertProfileType(p.type));

  if (p.data.empty()) {
    return out;
  }

  switch (p.type) {
    case SCAMP::PROFILE_TYPE_1NN_INDEX:
      if (p.data.front().uint64_value.empty()) {
        out.mutable_data()->Add();
        return out;
      }
      *out.mutable_data()->Add()->mutable_uint64_value()->mutable_value() = {
          p.data[0].uint64_value.begin(), p.data[0].uint64_value.end()};
      break;
    case SCAMP::PROFILE_TYPE_1NN:
      if (p.data.front().float_value.empty()) {
        out.mutable_data()->Add();
        return out;
      }
      *out.mutable_data()->Add()->mutable_float_value()->mutable_value() = {
          p.data[0].float_value.begin(), p.data[0].float_value.end()};
      break;
    case SCAMP::PROFILE_TYPE_SUM_THRESH:
      if (p.data.front().double_value.empty()) {
        out.mutable_data()->Add();
        return out;
      }
      *out.mutable_data()->Add()->mutable_double_value()->mutable_value() = {
          p.data[0].double_value.begin(), p.data[0].double_value.end()};
      break;
    default:
      ASSERT(false, "Undefined Profile Type");
      break;
  }
  return out;
}

// Converts between SCAMP:: and SCAMPProto:: profile formats
SCAMP::Profile ConvertProfile(const SCAMPProto::Profile &profile) {
  SCAMP::Profile out;
  out.type = ConvertProfileType(profile.type());
  if (profile.data().empty()) {
    return out;
  }
  switch (profile.type()) {
    case SCAMPProto::PROFILE_TYPE_1NN_INDEX:
      out.data.emplace_back();
      if (profile.data().Get(0).uint64_value().value().empty()) {
        return out;
      }
      out.data[0].uint64_value = {
          profile.data().Get(0).uint64_value().value().begin(),
          profile.data().Get(0).uint64_value().value().end()};
      return out;
    case SCAMPProto::PROFILE_TYPE_1NN:
      out.data.emplace_back();
      if (profile.data().Get(0).float_value().value().empty()) {
        return out;
      }
      out.data[0].float_value = {
          profile.data().Get(0).float_value().value().begin(),
          profile.data().Get(0).float_value().value().end()};
      return out;
    case SCAMPProto::PROFILE_TYPE_SUM_THRESH:
      out.data.emplace_back();
      if (profile.data().Get(0).double_value().value().empty()) {
        return out;
      }
      out.data[0].double_value = {
          profile.data().Get(0).double_value().value().begin(),
          profile.data().Get(0).double_value().value().end()};
      return out;
    default:
      ASSERT(false, "Undefined Profile Type");
      return out;
  }
  return out;
}

SCAMPProto::SCAMPArgs ConvertSCAMPArgsToProtoArgs(
    const SCAMP::SCAMPArgs &args) {
  SCAMPProto::SCAMPArgs proto_args;
  if (!args.timeseries_a.empty()) {
    *proto_args.mutable_timeseries_a() = {args.timeseries_a.begin(),
                                          args.timeseries_a.end()};
  }
  if (!args.timeseries_b.empty()) {
    *proto_args.mutable_timeseries_b() = {args.timeseries_b.begin(),
                                          args.timeseries_b.end()};
  }

  *proto_args.mutable_profile_a() = ConvertProfile(args.profile_a);
  *proto_args.mutable_profile_b() = ConvertProfile(args.profile_b);
  proto_args.set_has_b(args.has_b);
  proto_args.set_window(args.window);
  proto_args.set_max_tile_size(args.max_tile_size);
  proto_args.set_distributed_start_row(args.distributed_start_row);
  proto_args.set_distributed_start_col(args.distributed_start_col);
  proto_args.set_timeseries_size_a(args.timeseries_a.size());
  proto_args.set_timeseries_size_b(args.timeseries_b.size());
  proto_args.set_profile_type(ConvertProfileType(args.profile_type));
  proto_args.set_precision_type(ConvertPrecisionType(args.precision_type));
  proto_args.set_distance_threshold(args.distance_threshold);
  proto_args.set_computing_rows(args.computing_rows);
  proto_args.set_computing_columns(args.computing_columns);
  proto_args.set_keep_rows_separate(args.keep_rows_separate);
  proto_args.set_is_aligned(args.is_aligned);

  return proto_args;
}

void ConvertProtoArgsToSCAMPArgs(const SCAMPProto::SCAMPArgs &proto_args,
                                 SCAMP::SCAMPArgs *args) {
  std::vector<double> Ta_h, Tb_h;

  for (const double elem : proto_args.timeseries_a()) {
    Ta_h.push_back(elem);
  }

  for (const double elem : proto_args.timeseries_b()) {
    Tb_h.push_back(elem);
  }

  args->max_tile_size = proto_args.max_tile_size();
  args->distributed_start_row = proto_args.distributed_start_row();
  args->distributed_start_col = proto_args.distributed_start_col();
  args->distance_threshold = proto_args.distance_threshold();
  args->computing_columns = proto_args.computing_columns();
  args->computing_rows = proto_args.computing_rows();
  args->profile_a = ConvertProfile(proto_args.profile_a());
  if (proto_args.keep_rows_separate()) {
    args->profile_b = ConvertProfile(proto_args.profile_b());
  }
  args->profile_type = ConvertProfileType(proto_args.profile_type());
  args->precision_type = ConvertPrecisionType(proto_args.precision_type());
  args->keep_rows_separate = proto_args.keep_rows_separate();
  args->is_aligned = proto_args.is_aligned();
  args->window = proto_args.window();
  args->has_b = proto_args.has_b();
  args->timeseries_a = std::move(Ta_h);
  args->timeseries_b = std::move(Tb_h);
  args->silent_mode = true;
}

int64_t GetProfileSize(const SCAMPProto::Profile &p) {
  if (p.data().empty()) {
    return 0;
  }
  switch (p.type()) {
    case SCAMPProto::PROFILE_TYPE_1NN_INDEX:
      return p.data(0).uint64_value().value_size();
    case SCAMPProto::PROFILE_TYPE_1NN:
      return p.data(0).float_value().value_size();
    case SCAMPProto::PROFILE_TYPE_SUM_THRESH:
      return p.data(0).double_value().value_size();
    default:
      ASSERT(false, "Undefined Profile Type");
      return -1;
  }
  return 0;
}

std::vector<double> GenerateRandomVector(int NumberCount, int minimum,
                                         int maximum) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::vector<double> values(NumberCount);
  std::uniform_real_distribution<> dis(minimum, maximum);
  std::generate(values.begin(), values.end(), [&]() { return dis(gen); });
  return values;
}

SCAMP::SCAMPArgs get_default_args(uint64_t input_size) {
  SCAMP::SCAMPArgs args;
  args.timeseries_a = GenerateRandomVector(input_size, -1, 1);
  args.has_b = false;
  args.window = 100;
  args.max_tile_size = 131072;
  args.distributed_start_row = -1;
  args.distributed_start_col = -1;
  args.distance_threshold = NAN;
  args.computing_rows = true;
  args.computing_columns = true;
  args.keep_rows_separate = false;
  args.is_aligned = false;
  args.silent_mode = true;
  args.precision_type = SCAMP::PRECISION_DOUBLE;
  args.profile_type = SCAMP::PROFILE_TYPE_1NN_INDEX;
  args.profile_a.type = SCAMP::PROFILE_TYPE_1NN_INDEX;
  args.profile_b.type = SCAMP::PROFILE_TYPE_1NN_INDEX;
  return args;
}

bool InitProfile(SCAMPProto::Profile *p, SCAMPProto::SCAMPProfileType type,
                 int64_t size) {
  if (size <= 0) {
    return false;
  }
  p->set_type(type);
  switch (type) {
    case SCAMPProto::PROFILE_TYPE_1NN_INDEX: {
      SCAMP::mp_entry e;
      e.floats[0] = -2;
      e.ints[1] = -1;
      std::vector<uint64_t> v(size, e.ulong);
      *p->mutable_data()->Add()->mutable_uint64_value()->mutable_value() = {
          v.begin(), v.end()};
      return true;
    }
    case SCAMPProto::PROFILE_TYPE_1NN: {
      std::vector<float> v(size, -2);
      *p->mutable_data()->Add()->mutable_float_value()->mutable_value() = {
          v.begin(), v.end()};
      return true;
    }
    case SCAMPProto::PROFILE_TYPE_SUM_THRESH: {
      std::vector<double> v(size, 0);
      *p->mutable_data()->Add()->mutable_double_value()->mutable_value() = {
          v.begin(), v.end()};
      return true;
    }
    default:
      return false;
  }
}

bool ProfileAllocated(const SCAMPProto::Profile &p) {
  if (p.data_size() <= 0) {
    return false;
  }
  switch (p.type()) {
    case SCAMPProto::PROFILE_TYPE_1NN_INDEX:
      return !p.data(0).uint64_value().value().empty();
    case SCAMPProto::PROFILE_TYPE_1NN:
      return !p.data(0).float_value().value().empty();
    case SCAMPProto::PROFILE_TYPE_SUM_THRESH:
      return !p.data(0).double_value().value().empty();
    default:
      return false;
  }
}

template <typename T>
void elementwise_sum(T *mp_full, uint64_t merge_start, uint64_t tile_sz,
                     T *to_merge) {
  for (int i = 0; i < tile_sz; ++i) {
    mp_full[i + merge_start] += to_merge[i];
  }
}

template <typename T>
void elementwise_max(T *mp_full, uint64_t merge_start, uint64_t tile_sz,
                     T *to_merge, uint64_t index_offset) {
  for (int i = 0; i < tile_sz; ++i) {
    SCAMP::mp_entry e1, e2;
    e1.ulong = mp_full[i + merge_start];
    e2.ulong = to_merge[i];
    if (e1.floats[0] < e2.floats[0]) {
      e2.ints[1] += index_offset;
      mp_full[i + merge_start] = e2.ulong;
    }
  }
}

template <typename T>
void elementwise_max(T *mp_full, uint64_t merge_start, uint64_t tile_sz,
                     T *to_merge) {
  for (int i = 0; i < tile_sz; ++i) {
    if (mp_full[i + merge_start] < to_merge[i]) {
      mp_full[i + merge_start] = to_merge[i];
    }
  }
}

void MergeTileIntoFullProfile(SCAMPProto::Profile *tile_profile,
                              uint64_t position, uint64_t length,
                              SCAMPProto::Profile *full_profile,
                              uint64_t index_start) {
  switch (full_profile->type()) {
    case SCAMPProto::PROFILE_TYPE_SUM_THRESH:
      elementwise_sum<double>(full_profile->mutable_data(0)
                                  ->mutable_double_value()
                                  ->mutable_value()
                                  ->mutable_data(),
                              position, length,
                              tile_profile->mutable_data(0)
                                  ->mutable_double_value()
                                  ->mutable_value()
                                  ->mutable_data());
      return;
    case SCAMPProto::PROFILE_TYPE_1NN_INDEX:
      elementwise_max<uint64_t>(full_profile->mutable_data(0)
                                    ->mutable_uint64_value()
                                    ->mutable_value()
                                    ->mutable_data(),
                                position, length,
                                tile_profile->mutable_data(0)
                                    ->mutable_uint64_value()
                                    ->mutable_value()
                                    ->mutable_data(),
                                index_start);
      return;
    case SCAMPProto::PROFILE_TYPE_1NN:
      elementwise_max<float>(full_profile->mutable_data(0)
                                 ->mutable_float_value()
                                 ->mutable_value()
                                 ->mutable_data(),
                             position, length,
                             tile_profile->mutable_data(0)
                                 ->mutable_float_value()
                                 ->mutable_value()
                                 ->mutable_data());
      return;
    case SCAMPProto::PROFILE_TYPE_FREQUENCY_THRESH:
    case SCAMPProto::PROFILE_TYPE_KNN:
    case SCAMPProto::PROFILE_TYPE_1NN_MULTIDIM:
    default:
      ASSERT(false, "FUNCTIONALITY UNIMPLEMENTED");
      return;
  }
}

void MergeProfile(const SCAMPProto::SCAMPArgs &tile_args,
                  SCAMPProto::SCAMPArgs *job_args, SCAMPProto::Profile *a_tile,
                  uint64_t col_pos, uint64_t width, SCAMPProto::Profile *b_tile,
                  uint64_t row_pos, uint64_t height) {
  // Merge result
  MergeTileIntoFullProfile(a_tile, col_pos, width,
                           job_args->mutable_profile_a(), row_pos);
  if (tile_args.keep_rows_separate()) {
    if (job_args->computing_rows() && job_args->keep_rows_separate()) {
      MergeTileIntoFullProfile(b_tile, row_pos, height,
                               job_args->mutable_profile_b(), col_pos);
    } else if (!job_args->has_b()) {
      MergeTileIntoFullProfile(b_tile, row_pos, height,
                               job_args->mutable_profile_a(), col_pos);
    }
  }
}

// TODO(zpzim): finish this stub
bool validateArgs(const SCAMPProto::SCAMPArgs &args) { return true; }  // NOLINT
