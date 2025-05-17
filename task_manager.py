import time
import uuid
import logging
import json
import os
from typing import Dict, List, Any, Optional, Callable, Tuple
from enum import Enum, auto
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TaskManager")

class TaskStatus(Enum):
    """Enum for task status."""
    PENDING = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()

class TaskPriority(Enum):
    """Enum for task priority."""
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()

class Task:
    """Represents a task to be executed by the agent."""
    
    def __init__(self, 
                 name: str, 
                 description: str, 
                 action: str,
                 params: Dict[str, Any] = None,
                 priority: TaskPriority = TaskPriority.MEDIUM,
                 deadline: Optional[datetime] = None,
                 dependencies: List[str] = None):
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.action = action
        self.params = params or {}
        self.priority = priority
        self.status = TaskStatus.PENDING
        self.created_at = datetime.now()
        self.started_at = None
        self.completed_at = None
        self.deadline = deadline
        self.dependencies = dependencies or []
        self.result = None
        self.error = None
        self.progress = 0.0  # 0.0 to 1.0
        self.subtasks = []
        self.parent_id = None
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "action": self.action,
            "params": self.params,
            "priority": self.priority.name,
            "status": self.status.name,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "dependencies": self.dependencies,
            "result": self.result,
            "error": self.error,
            "progress": self.progress,
            "subtasks": [subtask.id for subtask in self.subtasks],
            "parent_id": self.parent_id
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        """Create task from dictionary."""
        task = cls(
            name=data["name"],
            description=data["description"],
            action=data["action"],
            params=data["params"],
            priority=TaskPriority[data["priority"]],
            deadline=datetime.fromisoformat(data["deadline"]) if data.get("deadline") else None,
            dependencies=data.get("dependencies", [])
        )
        task.id = data["id"]
        task.status = TaskStatus[data["status"]]
        task.created_at = datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None
        task.started_at = datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None
        task.completed_at = datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None
        task.result = data.get("result")
        task.error = data.get("error")
        task.progress = data.get("progress", 0.0)
        task.parent_id = data.get("parent_id")
        # Note: subtasks are handled separately by the TaskManager
        return task
        
    def start(self) -> None:
        """Mark the task as started."""
        self.status = TaskStatus.IN_PROGRESS
        self.started_at = datetime.now()
        
    def complete(self, result: Any = None) -> None:
        """Mark the task as completed."""
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.now()
        self.result = result
        self.progress = 1.0
        
    def fail(self, error: str) -> None:
        """Mark the task as failed."""
        self.status = TaskStatus.FAILED
        self.completed_at = datetime.now()
        self.error = error
        
    def cancel(self) -> None:
        """Mark the task as cancelled."""
        self.status = TaskStatus.CANCELLED
        self.completed_at = datetime.now()
        
    def update_progress(self, progress: float) -> None:
        """Update the progress of the task."""
        self.progress = max(0.0, min(1.0, progress))  # Ensure progress is between 0 and 1
        
    def add_subtask(self, subtask: 'Task') -> None:
        """Add a subtask to this task."""
        subtask.parent_id = self.id
        self.subtasks.append(subtask)
        
    def is_overdue(self) -> bool:
        """Check if the task is overdue."""
        if not self.deadline:
            return False
        return datetime.now() > self.deadline
        
    def time_until_deadline(self) -> Optional[timedelta]:
        """Get the time until the deadline."""
        if not self.deadline:
            return None
        return self.deadline - datetime.now()
        
    def can_execute(self, completed_task_ids: List[str]) -> bool:
        """Check if the task can be executed based on dependencies."""
        return all(dep_id in completed_task_ids for dep_id in self.dependencies)


class TaskManager:
    """Manages tasks for the agent."""
    
    def __init__(self, storage_path: str = "tasks.json"):
        self.tasks: Dict[str, Task] = {}
        self.completed_task_ids: List[str] = []
        self.storage_path = storage_path
        self.action_handlers: Dict[str, Callable] = {}
        self.load()
        
    def register_action_handler(self, action: str, handler: Callable) -> None:
        """Register a handler for a specific action."""
        self.action_handlers[action] = handler
        
    def create_task(self, 
                   name: str, 
                   description: str, 
                   action: str,
                   params: Dict[str, Any] = None,
                   priority: TaskPriority = TaskPriority.MEDIUM,
                   deadline: Optional[datetime] = None,
                   dependencies: List[str] = None) -> Task:
        """Create a new task."""
        task = Task(
            name=name,
            description=description,
            action=action,
            params=params,
            priority=priority,
            deadline=deadline,
            dependencies=dependencies
        )
        self.tasks[task.id] = task
        self.save()
        return task
        
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        return self.tasks.get(task_id)
        
    def get_all_tasks(self) -> List[Task]:
        """Get all tasks."""
        return list(self.tasks.values())
        
    def get_pending_tasks(self) -> List[Task]:
        """Get all pending tasks."""
        return [task for task in self.tasks.values() if task.status == TaskStatus.PENDING]
        
    def get_in_progress_tasks(self) -> List[Task]:
        """Get all in-progress tasks."""
        return [task for task in self.tasks.values() if task.status == TaskStatus.IN_PROGRESS]
        
    def get_completed_tasks(self) -> List[Task]:
        """Get all completed tasks."""
        return [task for task in self.tasks.values() if task.status == TaskStatus.COMPLETED]
        
    def get_failed_tasks(self) -> List[Task]:
        """Get all failed tasks."""
        return [task for task in self.tasks.values() if task.status == TaskStatus.FAILED]
        
    def get_next_task(self) -> Optional[Task]:
        """Get the next task to execute based on priority and dependencies."""
        pending_tasks = self.get_pending_tasks()
        
        # Filter tasks that can be executed (dependencies satisfied)
        executable_tasks = [task for task in pending_tasks if task.can_execute(self.completed_task_ids)]
        
        if not executable_tasks:
            return None
            
        # Sort by priority (higher first) and then by deadline (earlier first)
        def task_sort_key(task):
            priority_value = -task.priority.value  # Negative to sort higher priority first
            deadline_value = task.deadline.timestamp() if task.deadline else float('inf')
            return (priority_value, deadline_value)
            
        executable_tasks.sort(key=task_sort_key)
        return executable_tasks[0]
        
    def execute_task(self, task_id: str) -> Tuple[bool, Any]:
        """Execute a task by ID."""
        task = self.get_task(task_id)
        if not task:
            return False, f"Task with ID {task_id} not found"
            
        if task.status != TaskStatus.PENDING:
            return False, f"Task is not pending (status: {task.status.name})"
            
        if not task.can_execute(self.completed_task_ids):
            return False, "Task dependencies not satisfied"
            
        # Mark task as in progress
        task.start()
        self.save()
        
        # Execute the task
        action = task.action
        if action not in self.action_handlers:
            task.fail(f"No handler registered for action: {action}")
            self.save()
            return False, f"No handler registered for action: {action}"
            
        try:
            handler = self.action_handlers[action]
            result = handler(task.params)
            task.complete(result)
            self.completed_task_ids.append(task.id)
            self.save()
            return True, result
        except Exception as e:
            error_message = str(e)
            task.fail(error_message)
            self.save()
            return False, error_message
            
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task by ID."""
        task = self.get_task(task_id)
        if not task:
            return False
            
        if task.status not in [TaskStatus.PENDING, TaskStatus.IN_PROGRESS]:
            return False
            
        task.cancel()
        self.save()
        return True
        
    def update_task_progress(self, task_id: str, progress: float) -> bool:
        """Update the progress of a task."""
        task = self.get_task(task_id)
        if not task:
            return False
            
        if task.status != TaskStatus.IN_PROGRESS:
            return False
            
        task.update_progress(progress)
        self.save()
        return True
        
    def create_subtask(self, 
                      parent_id: str,
                      name: str, 
                      description: str, 
                      action: str,
                      params: Dict[str, Any] = None,
                      priority: TaskPriority = TaskPriority.MEDIUM) -> Optional[Task]:
        """Create a subtask for a parent task."""
        parent_task = self.get_task(parent_id)
        if not parent_task:
            return None
            
        subtask = Task(
            name=name,
            description=description,
            action=action,
            params=params,
            priority=priority
        )
        subtask.parent_id = parent_id
        
        self.tasks[subtask.id] = subtask
        parent_task.add_subtask(subtask)
        self.save()
        return subtask
        
    def get_task_hierarchy(self, task_id: str) -> Dict[str, Any]:
        """Get a task and all its subtasks as a hierarchy."""
        task = self.get_task(task_id)
        if not task:
            return {}
            
        result = task.to_dict()
        result["subtasks"] = [self.get_task_hierarchy(subtask.id) for subtask in task.subtasks]
        return result
        
    def get_overdue_tasks(self) -> List[Task]:
        """Get all overdue tasks."""
        return [task for task in self.tasks.values() if task.is_overdue()]
        
    def clean_completed_tasks(self, days: int = 30) -> int:
        """Remove completed tasks older than the specified number of days."""
        cutoff_date = datetime.now() - timedelta(days=days)
        tasks_to_remove = []
        
        for task_id, task in self.tasks.items():
            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                if task.completed_at and task.completed_at < cutoff_date:
                    tasks_to_remove.append(task_id)
                    
        for task_id in tasks_to_remove:
            del self.tasks[task_id]
            if task_id in self.completed_task_ids:
                self.completed_task_ids.remove(task_id)
                
        self.save()
        return len(tasks_to_remove)
        
    def save(self) -> None:
        """Save tasks to disk."""
        try:
            data = {
                "tasks": {task_id: task.to_dict() for task_id, task in self.tasks.items()},
                "completed_task_ids": self.completed_task_ids
            }
            
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.info(f"Saved tasks to {self.storage_path}")
        except Exception as e:
            logger.error(f"Error saving tasks: {e}")
            
    def load(self) -> None:
        """Load tasks from disk."""
        if not os.path.exists(self.storage_path):
            logger.info(f"No tasks file found at {self.storage_path}")
            return
            
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                
            # First pass: create all tasks
            for task_id, task_data in data.get("tasks", {}).items():
                self.tasks[task_id] = Task.from_dict(task_data)
                
            # Second pass: set up subtask relationships
            for task_id, task_data in data.get("tasks", {}).items():
                task = self.tasks[task_id]
                for subtask_id in task_data.get("subtasks", []):
                    if subtask_id in self.tasks:
                        subtask = self.tasks[subtask_id]
                        task.subtasks.append(subtask)
                        
            self.completed_task_ids = data.get("completed_task_ids", [])
            
            logger.info(f"Loaded {len(self.tasks)} tasks from {self.storage_path}")
        except Exception as e:
            logger.error(f"Error loading tasks: {e}")


if __name__ == "__main__":
    # Example usage
    task_manager = TaskManager()
    
    # Register action handlers
    def send_email_handler(params):
        print(f"Sending email to {params.get('recipient')} with subject: {params.get('subject')}")
        return {"status": "sent"}
        
    def fetch_data_handler(params):
        print(f"Fetching data from {params.get('source')}")
        return {"data": [1, 2, 3, 4, 5]}
        
    task_manager.register_action_handler("send_email", send_email_handler)
    task_manager.register_action_handler("fetch_data", fetch_data_handler)
    
    # Create tasks
    fetch_task = task_manager.create_task(
        name="Fetch Data",
        description="Fetch data from the API",
        action="fetch_data",
        params={"source": "https://api.example.com/data"},
        priority=TaskPriority.HIGH
    )
    
    email_task = task_manager.create_task(
        name="Send Email",
        description="Send email with the fetched data",
        action="send_email",
        params={
            "recipient": "user@example.com",
            "subject": "Data Report",
            "body": "Here is the data you requested."
        },
        dependencies=[fetch_task.id]
    )
    
    # Execute tasks
    next_task = task_manager.get_next_task()
    while next_task:
        print(f"Executing task: {next_task.name}")
        success, result = task_manager.execute_task(next_task.id)
        print(f"Task {'succeeded' if success else 'failed'}: {result}")
        next_task = task_manager.get_next_task()