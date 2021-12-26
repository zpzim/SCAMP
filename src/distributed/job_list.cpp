#include "job_list.h"

uint64_t JobList::add_job(const SCAMPProto::SCAMPArgs& args) {
  std::lock_guard<std::mutex> lockGuard(task_list_mutex_);
  uint64_t job_id = task_list_.size();
  task_list_.emplace(job_id, Job(args, job_id));
  run_list_.emplace_back(job_id);
  return job_id;
}

Job* JobList::get_job(int job_id) {
  std::lock_guard<std::mutex> lockGuard(task_list_mutex_);
  if (task_list_.count(job_id) == 0) {
    return nullptr;
  }
  return &task_list_.at(job_id);
}

Job* JobList::get_job_to_work_on() {
  std::lock_guard<std::mutex> lockGuard(task_list_mutex_);
  return get_highest_priority_job();
}

void JobList::cleanup_jobs() {
  std::lock_guard<std::mutex> lockGuard(task_list_mutex_);
  std::vector<int> to_remove;
  auto iter = run_list_.begin();
  while (iter != run_list_.end()) {
    Job& job = task_list_.at(*iter);
    // Check if job timed out
    job.check_time_tile();
    if (job.status() != SCAMPProto::JOB_STATUS_RUNNING) {
      // Remove job from run_list if it has finished
      iter = run_list_.erase(iter);
    } else {
      iter++;
    }
  }
}

Job* JobList::ScheduleIssueOrder() {
  for (auto& job_id : run_list_) {
    Job& job = task_list_.at(job_id);
    if (job.has_work()) {
      return &job;
    }
  }
  return nullptr;
}

Job* JobList::ScheduleRoundRobin() {
  for (int i = 0; i < run_list_.size(); ++i) {
    int job_id = run_list_.front();
    run_list_.pop_front();
    run_list_.push_back(job_id);
    Job& job = task_list_.at(job_id);
    if (job.has_work()) {
      return &job;
    }
  }
  return nullptr;
}

Job* JobList::ScheduleEndingSoonest() {
  int best_eta = INT_MAX;
  Job* best_job = nullptr;
  for (auto& job_id : run_list_) {
    if (task_list_.at(job_id).has_work()) {
      int eta = task_list_.at(job_id).get_eta();
      if (eta < best_eta) {
        best_eta = eta;
        best_job = &task_list_.at(job_id);
      }
    }
  }
  return best_job;
}

Job* JobList::ScheduleLowestProgress() {
  double best_prog = 1.0;
  Job* best_job = nullptr;
  for (auto& job_id : run_list_) {
    if (task_list_.at(job_id).has_work()) {
      double prog = task_list_.at(job_id).get_progress();
      if (prog < best_prog) {
        best_prog = prog;
        best_job = &task_list_.at(job_id);
      }
    }
  }
  return best_job;
}

Job* JobList::get_highest_priority_job() {
  switch (schedule_type_) {
    case SCHEDULE_TYPE_ISSUE_ORDER:
      return ScheduleIssueOrder();
    case SCHEDULE_TYPE_ROUND_ROBIN:
      return ScheduleRoundRobin();
    case SCHEDULE_TYPE_LEAST_ETA:
      return ScheduleEndingSoonest();
    case SCHEDULE_TYPE_LEAST_PROGRESS:
      return ScheduleLowestProgress();
  }
  return nullptr;
}
