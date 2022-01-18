#pragma once

#include <list>
#include <mutex>
#include <unordered_map>

#include "distributed_job.h"
#include "scamp.pb.h"

enum JobListSchedulerType {
  SCHEDULE_TYPE_ISSUE_ORDER = 0,
  SCHEDULE_TYPE_ROUND_ROBIN = 1,
  SCHEDULE_TYPE_LEAST_ETA = 2,
  SCHEDULE_TYPE_LEAST_PROGRESS = 3,
};

class JobList {
 public:
  JobList(JobListSchedulerType t) : schedule_type_(t) {}
  uint64_t add_job(const SCAMPProto::SCAMPArgs& args);
  Job* get_job(int job_id);
  Job* get_job_to_work_on();
  void cleanup_jobs();

 private:
  // Schedule routines
  Job* ScheduleIssueOrder();
  Job* ScheduleRoundRobin();
  Job* ScheduleEndingSoonest();
  Job* ScheduleLowestProgress();

  // Schedule routine picker
  Job* get_highest_priority_job();

  // Mutex protecting access to a this task list
  std::mutex task_list_mutex_;
  // Task list
  std::unordered_map<int, Job> task_list_;
  // List of running tasks
  std::list<int> run_list_;
  // Type of scheduling used by the job list
  JobListSchedulerType schedule_type_;
};
