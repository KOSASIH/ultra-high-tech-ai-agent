from flask import Flask, render_template, request, jsonify
from autonomous_agent import AutonomousAgent
import logging
import os
import random
import json
import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AgentWebApp")

# Initialize the agent
agent = AutonomousAgent(name="UltraHighTechAI")

# Try to load previous state if it exists
state_path = "agent_state.json"
if os.path.exists(state_path):
    try:
        agent.load_state(state_path)
        logger.info("Loaded previous agent state")
    except Exception as e:
        logger.error(f"Failed to load previous state: {e}")

app = Flask(__name__)

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Process a chat message from the user."""
    data = request.json
    user_input = data.get('message', '')
    user_name = data.get('user_name', 'User')
    
    if not user_input:
        return jsonify({"error": "No message provided"}), 400
    
    # Process the input with our agent
    response = agent.process_input(user_input, user_name)
    
    # Periodically save state and learn
    if random.random() < 0.1:  # 10% chance after each message
        agent.learn()
        agent.save_state(state_path)
    
    return jsonify({
        "response": response,
        "agent_name": agent.name
    })

@app.route('/api/learn', methods=['POST'])
def trigger_learning():
    """Manually trigger the learning process."""
    agent.learn()
    agent.save_state(state_path)
    return jsonify({"status": "success", "message": "Learning completed"})

@app.route('/api/reset', methods=['POST'])
def reset_agent():
    """Reset the agent to its initial state."""
    global agent
    
    # Stop the task processor before resetting
    if hasattr(agent, 'stop_task_processor'):
        agent.stop_task_processor()
    
    agent = AutonomousAgent(name="UltraHighTechAI")
    if os.path.exists(state_path):
        try:
            os.remove(state_path)
        except Exception as e:
            logger.error(f"Failed to remove state file: {e}")
    
    # Remove knowledge base and task files
    for file_path in ["knowledge_base.json", "tasks.json"]:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Removed {file_path}")
            except Exception as e:
                logger.error(f"Failed to remove {file_path}: {e}")
    
    return jsonify({"status": "success", "message": "Agent reset to initial state"})

@app.route('/api/tasks', methods=['GET'])
def get_tasks():
    """Get all tasks."""
    if not hasattr(agent, 'get_all_tasks'):
        return jsonify({"error": "Task management not available"}), 400
    
    tasks = agent.get_all_tasks()
    return jsonify(tasks)

@app.route('/api/tasks', methods=['POST'])
def create_task():
    """Create a new task."""
    if not hasattr(agent, 'create_task'):
        return jsonify({"error": "Task management not available"}), 400
    
    data = request.json
    name = data.get('name', '')
    description = data.get('description', '')
    action = data.get('action', '')
    params = data.get('params', {})
    priority = data.get('priority', 'MEDIUM')
    deadline = data.get('deadline')
    
    if not name or not description or not action:
        return jsonify({"error": "Missing required fields"}), 400
    
    result = agent.create_task(
        name=name,
        description=description,
        action=action,
        params=params,
        priority=priority,
        deadline_str=deadline
    )
    
    return jsonify(result)

@app.route('/api/tasks/<task_id>', methods=['GET'])
def get_task(task_id):
    """Get a specific task."""
    if not hasattr(agent, 'get_task_status'):
        return jsonify({"error": "Task management not available"}), 400
    
    task = agent.get_task_status(task_id)
    
    if "error" in task:
        return jsonify({"error": task["error"]}), 404
    
    return jsonify(task)

@app.route('/api/tasks/<task_id>', methods=['DELETE'])
def cancel_task(task_id):
    """Cancel a task."""
    if not hasattr(agent, 'cancel_task'):
        return jsonify({"error": "Task management not available"}), 400
    
    result = agent.cancel_task(task_id)
    
    if "error" in result:
        return jsonify({"error": result["error"]}), 400
    
    return jsonify(result)

@app.route('/api/knowledge/search', methods=['GET'])
def search_knowledge():
    """Search the knowledge base."""
    if not hasattr(agent, 'search_knowledge'):
        return jsonify({"error": "Knowledge base not available"}), 400
    
    query = request.args.get('q', '')
    
    if not query:
        return jsonify({"error": "Missing query parameter"}), 400
    
    results = agent.search_knowledge(query)
    return jsonify(results)

@app.route('/api/knowledge/learn', methods=['POST'])
def learn_knowledge():
    """Learn new information."""
    if not hasattr(agent, 'learn_information'):
        return jsonify({"error": "Knowledge base not available"}), 400
    
    data = request.json
    text = data.get('text', '')
    category = data.get('category', 'general')
    
    if not text:
        return jsonify({"error": "Missing text parameter"}), 400
    
    result = agent.learn_information(text, category)
    return jsonify(result)

@app.route('/api/knowledge/fact', methods=['GET'])
def get_random_fact():
    """Get a random fact."""
    if not hasattr(agent, 'get_random_fact'):
        return jsonify({"error": "Knowledge base not available"}), 400
    
    fact = agent.get_random_fact()
    return jsonify({"fact": fact})

@app.route('/api/knowledge/define/<term>', methods=['GET'])
def get_definition(term):
    """Get the definition of a term."""
    if not hasattr(agent, 'get_definition'):
        return jsonify({"error": "Knowledge base not available"}), 400
    
    definition = agent.get_definition(term)
    return jsonify(definition)

@app.route('/api/agent/status', methods=['GET'])
def get_agent_status():
    """Get the status of the agent."""
    # Get conversation history
    history = agent.memory.retrieve_long_term("conversation_history") or []
    
    # Get creation time
    creation_time_str = agent.memory.retrieve_long_term("creation_time")
    creation_time = datetime.datetime.fromisoformat(creation_time_str) if creation_time_str else datetime.datetime.now()
    
    # Calculate age
    age = datetime.datetime.now() - creation_time
    
    # Get task counts if available
    task_counts = {}
    if hasattr(agent, 'get_all_tasks'):
        tasks = agent.get_all_tasks()
        task_counts = {
            "total": tasks.get("total", 0),
            "pending": tasks.get("pending", 0),
            "in_progress": tasks.get("in_progress", 0),
            "completed": tasks.get("completed", 0),
            "failed": tasks.get("failed", 0)
        }
    
    return jsonify({
        "name": agent.name,
        "creation_time": creation_time_str,
        "age_days": age.days,
        "age_hours": age.seconds // 3600,
        "conversation_count": len(history),
        "task_counts": task_counts,
        "capabilities": [
            "Natural language understanding",
            "Task management",
            "Knowledge base",
            "Learning from interactions",
            "Sentiment analysis"
        ]
    })

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=8080, debug=True)