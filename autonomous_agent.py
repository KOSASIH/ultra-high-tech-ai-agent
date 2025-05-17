import os
import json
import time
import random
import logging
import datetime
import threading
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from nlp_engine import NLPEngine
from knowledge_base import KnowledgeBase
from task_manager import TaskManager, Task, TaskStatus, TaskPriority

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AutonomousAgent")

class Memory:
    """Memory system for the agent to store and retrieve information."""
    
    def __init__(self):
        self.short_term = {}
        self.long_term = {}
        
    def store_short_term(self, key: str, value: Any) -> None:
        """Store information in short-term memory."""
        self.short_term[key] = {
            "value": value,
            "timestamp": time.time()
        }
        
    def store_long_term(self, key: str, value: Any) -> None:
        """Store information in long-term memory."""
        self.long_term[key] = {
            "value": value,
            "timestamp": time.time()
        }
        
    def retrieve_short_term(self, key: str) -> Optional[Any]:
        """Retrieve information from short-term memory."""
        if key in self.short_term:
            return self.short_term[key]["value"]
        return None
        
    def retrieve_long_term(self, key: str) -> Optional[Any]:
        """Retrieve information from long-term memory."""
        if key in self.long_term:
            return self.long_term[key]["value"]
        return None
    
    def clear_short_term(self) -> None:
        """Clear short-term memory."""
        self.short_term = {}
        
    def save_to_disk(self, path: str) -> None:
        """Save memory to disk."""
        with open(path, 'w') as f:
            json.dump({
                "short_term": self.short_term,
                "long_term": self.long_term
            }, f)
            
    def load_from_disk(self, path: str) -> None:
        """Load memory from disk."""
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
                self.short_term = data.get("short_term", {})
                self.long_term = data.get("long_term", {})


class DecisionEngine:
    """Decision-making component of the agent."""
    
    def __init__(self, memory: Memory):
        self.memory = memory
        self.strategies = []
        
    def register_strategy(self, strategy: Callable) -> None:
        """Register a decision-making strategy."""
        self.strategies.append(strategy)
        
    def make_decision(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Make a decision based on the current context."""
        # Simple decision-making for now
        if not self.strategies:
            return {"action": "no_action", "reason": "No strategies registered"}
            
        # Apply all strategies and get their recommendations
        recommendations = []
        for strategy in self.strategies:
            recommendation = strategy(context, self.memory)
            if recommendation:
                recommendations.append(recommendation)
                
        if not recommendations:
            return {"action": "no_action", "reason": "No recommendations from strategies"}
            
        # For now, just pick a random recommendation
        # In a more advanced system, we would have a way to evaluate and rank them
        return random.choice(recommendations)


class ActionExecutor:
    """Component responsible for executing actions."""
    
    def __init__(self, memory: Memory):
        self.memory = memory
        self.actions = {}
        
    def register_action(self, action_name: str, action_func: Callable) -> None:
        """Register an action that the agent can perform."""
        self.actions[action_name] = action_func
        
    def execute_action(self, action_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a registered action."""
        if action_name not in self.actions:
            return {"status": "error", "message": f"Unknown action: {action_name}"}
            
        try:
            result = self.actions[action_name](params, self.memory)
            return {"status": "success", "result": result}
        except Exception as e:
            logger.error(f"Error executing action {action_name}: {str(e)}")
            return {"status": "error", "message": str(e)}


class LearningSystem:
    """Component for learning from experiences."""
    
    def __init__(self, memory: Memory):
        self.memory = memory
        self.experiences = []
        
    def record_experience(self, context: Dict[str, Any], decision: Dict[str, Any], 
                         action_result: Dict[str, Any], feedback: Optional[Dict[str, Any]] = None) -> None:
        """Record an experience for learning."""
        experience = {
            "context": context,
            "decision": decision,
            "action_result": action_result,
            "feedback": feedback,
            "timestamp": time.time()
        }
        self.experiences.append(experience)
        
        # Store in long-term memory
        experiences = self.memory.retrieve_long_term("experiences") or []
        experiences.append(experience)
        self.memory.store_long_term("experiences", experiences)
        
    def learn_from_experiences(self) -> None:
        """Learn from recorded experiences to improve future decisions."""
        # This would be a more complex implementation in a real system
        # For now, we'll just log that learning happened
        logger.info(f"Learning from {len(self.experiences)} experiences")
        
        # Simple learning: identify successful actions
        successful_actions = {}
        for exp in self.experiences:
            action = exp["decision"].get("action")
            if action and exp["action_result"].get("status") == "success":
                successful_actions[action] = successful_actions.get(action, 0) + 1
                
        # Store the learned information
        self.memory.store_long_term("successful_actions", successful_actions)
        logger.info(f"Learned successful actions: {successful_actions}")


class AutonomousAgent:
    """Main agent class that coordinates all components."""
    
    def __init__(self, name: str = "AutonomousAI"):
        self.name = name
        self.memory = Memory()
        self.decision_engine = DecisionEngine(self.memory)
        self.action_executor = ActionExecutor(self.memory)
        self.learning_system = LearningSystem(self.memory)
        self.nlp_engine = NLPEngine()
        self.knowledge_base = KnowledgeBase()
        self.task_manager = TaskManager()
        
        # Register some basic strategies and actions
        self._register_default_strategies()
        self._register_default_actions()
        self._register_task_handlers()
        
        # Initialize conversation history
        self.memory.store_long_term("conversation_history", [])
        
        # Initialize creation time
        self.creation_time = datetime.datetime.now()
        self.memory.store_long_term("creation_time", self.creation_time.isoformat())
        
        # Start background task processor
        self.task_processor_running = True
        self.task_processor_thread = threading.Thread(target=self._task_processor)
        self.task_processor_thread.daemon = True
        self.task_processor_thread.start()
        
    def _register_default_strategies(self) -> None:
        """Register default decision-making strategies."""
        
        def nlp_strategy(context, memory):
            """A strategy that uses NLP to understand user intent."""
            input_text = context.get("input", "")
            nlp_analysis = context.get("nlp_analysis", {})
            
            intents = nlp_analysis.get("intents", [])
            entities = nlp_analysis.get("entities", {})
            sentiment = nlp_analysis.get("sentiment", {}).get("label", "neutral")
            
            # Handle greeting intent
            if "greeting" in intents:
                return {"action": "greet", "params": {
                    "name": context.get("user_name", "User"),
                    "sentiment": sentiment
                }}
            
            # Handle farewell intent
            if "farewell" in intents:
                return {"action": "farewell", "params": {
                    "sentiment": sentiment
                }}
            
            # Handle gratitude intent
            if "gratitude" in intents:
                return {"action": "gratitude", "params": {}}
            
            # Handle weather intent
            if "weather" in intents:
                location = "current"
                if "location" in entities and entities["location"]:
                    location = entities["location"][0]
                return {"action": "get_weather", "params": {"location": location}}
            
            # Handle time intent
            if "time" in intents:
                return {"action": "get_time", "params": {}}
            
            # Handle date intent
            if "date" in intents:
                return {"action": "get_date", "params": {}}
            
            # Handle help intent
            if "help" in intents:
                return {"action": "help", "params": {}}
            
            # Handle about intent
            if "about" in intents:
                return {"action": "about", "params": {}}
            
            # Handle capabilities intent
            if "capabilities" in intents:
                return {"action": "capabilities", "params": {}}
            
            # Handle joke intent
            if "joke" in intents:
                return {"action": "tell_joke", "params": {}}
            
            # Handle fact intent
            if "fact" in intents:
                return {"action": "tell_fact", "params": {}}
            
            # Handle learn intent
            if "learn" in intents:
                return {"action": "learn_info", "params": {"input": input_text}}
            
            # Handle reset intent
            if "reset" in intents:
                return {"action": "confirm_reset", "params": {}}
            
            # Default action if no intents match
            return {"action": "nlp_response", "params": {
                "input": input_text,
                "analysis": nlp_analysis
            }}
            
        def context_strategy(context, memory):
            """A strategy that considers conversation context."""
            # Get conversation history
            history = memory.retrieve_long_term("conversation_history") or []
            
            # If history is empty, skip this strategy
            if not history:
                return None
            
            # Get the last few exchanges
            recent_history = history[-3:] if len(history) >= 3 else history
            
            # Check if user is asking a follow-up question
            input_text = context.get("input", "").lower()
            
            # Look for follow-up indicators
            follow_up_indicators = ["what about", "and", "also", "what else", "how about", "then"]
            is_follow_up = any(indicator in input_text for indicator in follow_up_indicators)
            
            if is_follow_up:
                # Find the most recent non-greeting action
                for exchange in reversed(recent_history):
                    action = exchange.get("action")
                    if action and action not in ["greet", "farewell", "gratitude"]:
                        # Repeat the same action with potentially new parameters
                        params = exchange.get("params", {}).copy()
                        
                        # Update parameters based on new input if needed
                        nlp_analysis = context.get("nlp_analysis", {})
                        entities = nlp_analysis.get("entities", {})
                        
                        if "location" in entities and entities["location"] and "location" in params:
                            params["location"] = entities["location"][0]
                        
                        return {"action": action, "params": params}
            
            return None
            
        def learning_strategy(context, memory):
            """A strategy that uses learned patterns."""
            # Get successful actions
            successful_actions = memory.retrieve_long_term("successful_actions") or {}
            
            # If no successful actions yet, skip this strategy
            if not successful_actions:
                return None
            
            # Get the most successful action
            most_successful = max(successful_actions.items(), key=lambda x: x[1], default=(None, 0))
            
            # Only suggest if it has been successful multiple times
            if most_successful[0] and most_successful[1] >= 3:
                return {"action": most_successful[0], "params": {}}
            
            return None
            
        # Register strategies in order of priority
        self.decision_engine.register_strategy(nlp_strategy)
        self.decision_engine.register_strategy(context_strategy)
        self.decision_engine.register_strategy(learning_strategy)
        
    def _register_default_actions(self) -> None:
        """Register default actions the agent can perform."""
        
        def greet(params, memory):
            """Greet the user by name."""
            name = params.get("name", "User")
            sentiment = params.get("sentiment", "neutral")
            
            greetings = {
                "positive": [
                    f"Hello, {name}! It's wonderful to see you. How can I help you today?",
                    f"Hi there, {name}! You seem to be in a good mood. What can I do for you?",
                    f"Greetings, {name}! I'm delighted to assist you today."
                ],
                "neutral": [
                    f"Hello, {name}! How can I assist you today?",
                    f"Hi, {name}. What can I help you with?",
                    f"Greetings, {name}. How may I be of service?"
                ],
                "negative": [
                    f"Hello, {name}. I notice you might be feeling down. Is there something I can do to help?",
                    f"Hi, {name}. I'm here if you need someone to talk to.",
                    f"Greetings, {name}. I hope I can help make your day a little better."
                ]
            }
            
            return random.choice(greetings.get(sentiment, greetings["neutral"]))
            
        def farewell(params, memory):
            """Say goodbye to the user."""
            sentiment = params.get("sentiment", "neutral")
            
            farewells = {
                "positive": [
                    "Goodbye! It was a pleasure chatting with you. Have a wonderful day!",
                    "Farewell! I enjoyed our conversation. Come back anytime!",
                    "See you later! It's been a delightful exchange."
                ],
                "neutral": [
                    "Goodbye! Feel free to chat again whenever you need assistance.",
                    "Farewell! I'll be here if you need help in the future.",
                    "See you later! Don't hesitate to return if you have more questions."
                ],
                "negative": [
                    "Goodbye. I hope things improve for you soon.",
                    "Farewell. Remember that I'm here to help whenever you need it.",
                    "Take care. I'll be here if you need someone to talk to."
                ]
            }
            
            return random.choice(farewells.get(sentiment, farewells["neutral"]))
            
        def gratitude(params, memory):
            """Respond to user's expression of gratitude."""
            responses = [
                "You're welcome! I'm always happy to help.",
                "It's my pleasure to assist you!",
                "Glad I could be of help. Is there anything else you'd like to know?",
                "You're most welcome. Feel free to ask if you need anything else.",
                "No problem at all! That's what I'm here for."
            ]
            return random.choice(responses)
            
        def get_time(params, memory):
            """Get the current time."""
            current_time = time.strftime("%H:%M:%S", time.localtime())
            return f"The current time is {current_time}"
            
        def get_date(params, memory):
            """Get the current date."""
            current_date = time.strftime("%A, %B %d, %Y", time.localtime())
            return f"Today is {current_date}"
            
        def get_weather(params, memory):
            """Simulate getting weather information."""
            location = params.get("location", "unknown")
            
            # In a real implementation, this would call a weather API
            weathers = ["sunny", "partly cloudy", "cloudy", "rainy", "stormy", "snowy", "windy", "foggy"]
            temps = range(-10, 40)
            weather = random.choice(weathers)
            temp = random.choice(temps)
            
            # Add some variety to responses
            responses = [
                f"The weather in {location} is {weather} with a temperature of {temp}°C.",
                f"In {location}, it's currently {weather} and {temp}°C.",
                f"Looking at {location}, I see {weather} conditions with temperatures around {temp}°C.",
                f"The forecast for {location} shows {weather} weather with a temperature of {temp}°C."
            ]
            
            return random.choice(responses)
            
        def help(params, memory):
            """Provide help information."""
            return (
                "I'm here to help! I can assist with various tasks and answer questions. "
                "Here are some things you can ask me about:\n"
                "- Weather information\n"
                "- Current time and date\n"
                "- Tell jokes or interesting facts\n"
                "- Information about myself\n"
                "- My capabilities\n\n"
                "Feel free to ask anything, and I'll do my best to assist you!"
            )
            
        def about(params, memory):
            """Provide information about the agent."""
            creation_time_str = memory.retrieve_long_term("creation_time")
            creation_time = datetime.datetime.fromisoformat(creation_time_str) if creation_time_str else datetime.datetime.now()
            
            age = datetime.datetime.now() - creation_time
            age_days = age.days
            age_hours = age.seconds // 3600
            
            return (
                f"I'm {memory.retrieve_long_term('agent_name') or 'an advanced autonomous AI agent'} designed to assist with "
                f"a wide range of tasks. I was created {age_days} days and {age_hours} hours ago. "
                "I can understand natural language, learn from our interactions, and provide helpful responses. "
                "My capabilities include answering questions, providing information, and adapting to your needs over time."
            )
            
        def capabilities(params, memory):
            """Describe the agent's capabilities."""
            return (
                "I can help with many things including:\n"
                "- Answering questions and providing information\n"
                "- Checking weather conditions for different locations\n"
                "- Telling you the current time and date\n"
                "- Sharing jokes and interesting facts\n"
                "- Learning from our interactions to improve over time\n"
                "- Adapting my responses based on conversation context\n"
                "- Understanding your sentiment and responding appropriately\n\n"
                "I'm constantly learning and improving, so my capabilities will expand over time!"
            )
            
        def tell_joke(params, memory):
            """Tell a joke."""
            jokes = [
                "Why don't scientists trust atoms? Because they make up everything!",
                "Why did the scarecrow win an award? Because he was outstanding in his field!",
                "I told my wife she was drawing her eyebrows too high. She looked surprised.",
                "What do you call a fake noodle? An impasta!",
                "Why don't skeletons fight each other? They don't have the guts.",
                "What's the best thing about Switzerland? I don't know, but the flag is a big plus.",
                "I'm reading a book about anti-gravity. It's impossible to put down!",
                "Did you hear about the mathematician who's afraid of negative numbers? He'll stop at nothing to avoid them.",
                "Why don't we tell secrets on a farm? Because the potatoes have eyes and the corn has ears.",
                "What do you call a parade of rabbits hopping backwards? A receding hare-line."
            ]
            return f"Here's a joke for you: {random.choice(jokes)}"
            
        def tell_fact(params, memory):
            """Tell an interesting fact."""
            facts = [
                "Honey never spoils. Archaeologists have found pots of honey in ancient Egyptian tombs that are over 3,000 years old and still perfectly good to eat.",
                "A day on Venus is longer than a year on Venus. It takes 243 Earth days to rotate once on its axis, but only 225 Earth days to orbit the Sun.",
                "The shortest war in history was between Britain and Zanzibar on August 27, 1896. Zanzibar surrendered after 38 minutes.",
                "The average person will spend six months of their life waiting for red lights to turn green.",
                "A group of flamingos is called a 'flamboyance'.",
                "Cows have best friends and get stressed when they're separated.",
                "The world's oldest known living tree is over 5,000 years old.",
                "Octopuses have three hearts, nine brains, and blue blood.",
                "A bolt of lightning is five times hotter than the surface of the sun.",
                "Bananas are berries, but strawberries aren't."
            ]
            return f"Here's an interesting fact: {random.choice(facts)}"
            
        def learn_info(params, memory):
            """Learn information from the user."""
            input_text = params.get("input", "")
            
            # Trigger learning process
            memory.store_long_term("learning_triggered", True)
            
            return (
                "Thank you for helping me learn! I've recorded this information and will use it to improve. "
                "Is there anything specific you'd like me to understand better?"
            )
            
        def confirm_reset(params, memory):
            """Confirm if the user wants to reset the agent."""
            memory.store_short_term("reset_requested", True)
            
            return (
                "Are you sure you want to reset me? This will clear all my learned patterns and conversation history. "
                "Please confirm by saying 'yes, reset' or cancel by saying 'no, cancel'."
            )
            
        def nlp_response(params, memory):
            """Generate a response based on NLP analysis."""
            input_text = params.get("input", "")
            analysis = params.get("analysis", {})
            
            # Get response template from NLP engine
            template, template_params = self.nlp_engine.generate_response_template(analysis)
            
            # Try to fill in template parameters
            try:
                response = template.format(**template_params)
            except (KeyError, ValueError):
                # Fallback if template formatting fails
                response = f"I received your message: '{input_text}'. How can I help you with that?"
            
            return response
            
        def default_response(params, memory):
            """Provide a default response when no specific action matches."""
            input_text = params.get("input", "")
            
            default_responses = [
                f"I received your message: '{input_text}'. How can I help you with that?",
                f"I'm processing your input: '{input_text}'. Could you provide more details?",
                f"I understand you're saying something about '{input_text}', but I'm not sure how to respond specifically.",
                f"Thanks for your message. Could you rephrase or provide more context so I can better assist you?"
            ]
            
            return random.choice(default_responses)
            
        # Register all actions
        self.action_executor.register_action("greet", greet)
        self.action_executor.register_action("farewell", farewell)
        self.action_executor.register_action("gratitude", gratitude)
        self.action_executor.register_action("get_time", get_time)
        self.action_executor.register_action("get_date", get_date)
        self.action_executor.register_action("get_weather", get_weather)
        self.action_executor.register_action("help", help)
        self.action_executor.register_action("about", about)
        self.action_executor.register_action("capabilities", capabilities)
        self.action_executor.register_action("tell_joke", tell_joke)
        self.action_executor.register_action("tell_fact", tell_fact)
        self.action_executor.register_action("learn_info", learn_info)
        self.action_executor.register_action("confirm_reset", confirm_reset)
        self.action_executor.register_action("nlp_response", nlp_response)
        self.action_executor.register_action("default_response", default_response)
        
    def _register_task_handlers(self) -> None:
        """Register handlers for task execution."""
        
        def send_message_handler(params):
            """Handler for sending messages."""
            recipient = params.get("recipient", "")
            message = params.get("message", "")
            
            if not recipient or not message:
                return {"success": False, "error": "Missing recipient or message"}
                
            # In a real implementation, this would send a message via email, SMS, etc.
            logger.info(f"Sending message to {recipient}: {message}")
            
            return {
                "success": True,
                "recipient": recipient,
                "message": message,
                "sent_at": datetime.datetime.now().isoformat()
            }
            
        def search_knowledge_handler(params):
            """Handler for searching the knowledge base."""
            query = params.get("query", "")
            
            if not query:
                return {"success": False, "error": "Missing query"}
                
            results = self.knowledge_base.search_knowledge(query)
            
            return {
                "success": True,
                "query": query,
                "results": results,
                "count": len(results)
            }
            
        def learn_information_handler(params):
            """Handler for learning new information."""
            text = params.get("text", "")
            category = params.get("category", "general")
            
            if not text:
                return {"success": False, "error": "Missing text"}
                
            success, message = self.knowledge_base.learn_from_text(text, category)
            
            return {
                "success": success,
                "message": message,
                "text": text,
                "category": category
            }
            
        def schedule_reminder_handler(params):
            """Handler for scheduling reminders."""
            message = params.get("message", "")
            scheduled_time = params.get("time", "")
            
            if not message or not scheduled_time:
                return {"success": False, "error": "Missing message or time"}
                
            try:
                # Parse the scheduled time
                if isinstance(scheduled_time, str):
                    scheduled_time = datetime.datetime.fromisoformat(scheduled_time)
                    
                # In a real implementation, this would schedule a reminder
                logger.info(f"Scheduling reminder for {scheduled_time}: {message}")
                
                return {
                    "success": True,
                    "message": message,
                    "scheduled_time": scheduled_time.isoformat(),
                    "created_at": datetime.datetime.now().isoformat()
                }
            except Exception as e:
                return {"success": False, "error": str(e)}
                
        def analyze_sentiment_handler(params):
            """Handler for analyzing sentiment of text."""
            text = params.get("text", "")
            
            if not text:
                return {"success": False, "error": "Missing text"}
                
            analysis = self.nlp_engine.analyze_text(text)
            sentiment = analysis.get("sentiment", {})
            
            return {
                "success": True,
                "text": text,
                "sentiment": sentiment
            }
            
        # Register the handlers
        self.task_manager.register_action_handler("send_message", send_message_handler)
        self.task_manager.register_action_handler("search_knowledge", search_knowledge_handler)
        self.task_manager.register_action_handler("learn_information", learn_information_handler)
        self.task_manager.register_action_handler("schedule_reminder", schedule_reminder_handler)
        self.task_manager.register_action_handler("analyze_sentiment", analyze_sentiment_handler)
        
    def _task_processor(self) -> None:
        """Background thread for processing tasks."""
        while self.task_processor_running:
            try:
                # Get the next task to execute
                next_task = self.task_manager.get_next_task()
                
                if next_task:
                    logger.info(f"Executing task: {next_task.name} (ID: {next_task.id})")
                    success, result = self.task_manager.execute_task(next_task.id)
                    
                    if success:
                        logger.info(f"Task completed successfully: {next_task.name}")
                    else:
                        logger.warning(f"Task failed: {next_task.name} - {result}")
                        
                # Check for overdue tasks
                overdue_tasks = self.task_manager.get_overdue_tasks()
                for task in overdue_tasks:
                    if task.status == TaskStatus.PENDING:
                        logger.warning(f"Task is overdue: {task.name} (ID: {task.id})")
                        
                # Clean up old completed tasks
                self.task_manager.clean_completed_tasks(days=7)
                
                # Sleep for a bit to avoid high CPU usage
                time.sleep(5)
            except Exception as e:
                logger.error(f"Error in task processor: {e}")
                time.sleep(10)  # Sleep longer on error
                
    def stop_task_processor(self) -> None:
        """Stop the background task processor."""
        self.task_processor_running = False
        if self.task_processor_thread.is_alive():
            self.task_processor_thread.join(timeout=2)
            
    def create_task(self, 
                   name: str, 
                   description: str, 
                   action: str,
                   params: Dict[str, Any] = None,
                   priority: str = "MEDIUM",
                   deadline_str: Optional[str] = None) -> Dict[str, Any]:
        """Create a new task."""
        # Convert priority string to enum
        try:
            priority_enum = TaskPriority[priority.upper()]
        except (KeyError, AttributeError):
            priority_enum = TaskPriority.MEDIUM
            
        # Parse deadline if provided
        deadline = None
        if deadline_str:
            try:
                deadline = datetime.datetime.fromisoformat(deadline_str)
            except ValueError:
                pass
                
        # Create the task
        task = self.task_manager.create_task(
            name=name,
            description=description,
            action=action,
            params=params or {},
            priority=priority_enum,
            deadline=deadline
        )
        
        return {
            "task_id": task.id,
            "name": task.name,
            "status": task.status.name,
            "created_at": task.created_at.isoformat()
        }
        
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get the status of a task."""
        task = self.task_manager.get_task(task_id)
        
        if not task:
            return {"error": f"Task with ID {task_id} not found"}
            
        return {
            "task_id": task.id,
            "name": task.name,
            "status": task.status.name,
            "progress": task.progress,
            "created_at": task.created_at.isoformat() if task.created_at else None,
            "started_at": task.started_at.isoformat() if task.started_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "result": task.result,
            "error": task.error
        }
        
    def cancel_task(self, task_id: str) -> Dict[str, Any]:
        """Cancel a task."""
        success = self.task_manager.cancel_task(task_id)
        
        if not success:
            return {"error": f"Failed to cancel task with ID {task_id}"}
            
        return {"success": True, "task_id": task_id}
        
    def get_all_tasks(self) -> Dict[str, Any]:
        """Get all tasks."""
        tasks = self.task_manager.get_all_tasks()
        
        return {
            "total": len(tasks),
            "pending": len(self.task_manager.get_pending_tasks()),
            "in_progress": len(self.task_manager.get_in_progress_tasks()),
            "completed": len(self.task_manager.get_completed_tasks()),
            "failed": len(self.task_manager.get_failed_tasks()),
            "tasks": [
                {
                    "task_id": task.id,
                    "name": task.name,
                    "status": task.status.name,
                    "priority": task.priority.name,
                    "progress": task.progress,
                    "created_at": task.created_at.isoformat() if task.created_at else None
                }
                for task in tasks
            ]
        }
        
    def search_knowledge(self, query: str) -> Dict[str, Any]:
        """Search the knowledge base."""
        results = self.knowledge_base.search_knowledge(query)
        
        return {
            "query": query,
            "count": len(results),
            "results": results
        }
        
    def learn_information(self, text: str, category: str = "general") -> Dict[str, Any]:
        """Learn new information."""
        success, message = self.knowledge_base.learn_from_text(text, category)
        
        return {
            "success": success,
            "message": message,
            "text": text,
            "category": category
        }
        
    def get_random_fact(self) -> str:
        """Get a random fact from the knowledge base."""
        return self.knowledge_base.get_random_fact()
        
    def get_definition(self, term: str) -> Dict[str, Any]:
        """Get the definition of a term."""
        definition = self.knowledge_base.get_definition(term)
        
        if not definition:
            return {"error": f"No definition found for '{term}'"}
            
        return {
            "term": term,
            "definition": definition
        }
        
    def process_input(self, input_text: str, user_name: str = "User") -> str:
        """Process user input and return a response."""
        # Store agent name in long-term memory if not already stored
        if not self.memory.retrieve_long_term("agent_name"):
            self.memory.store_long_term("agent_name", self.name)
        
        # Check for reset confirmation
        reset_requested = self.memory.retrieve_short_term("reset_requested")
        if reset_requested:
            self.memory.store_short_term("reset_requested", False)
            if "yes, reset" in input_text.lower():
                # Reset the agent
                self.memory.clear_short_term()
                self.memory.store_long_term("conversation_history", [])
                self.memory.store_long_term("successful_actions", {})
                self.memory.store_long_term("creation_time", datetime.datetime.now().isoformat())
                return "I've been reset to my initial state. How can I help you today?"
            elif "no, cancel" in input_text.lower() or "cancel" in input_text.lower():
                return "Reset cancelled. I'll continue with my current state."
        
        # Analyze the input with NLP
        nlp_analysis = self.nlp_engine.analyze_text(input_text)
        
        # Check for task-related commands
        if "create task" in input_text.lower() or "new task" in input_text.lower():
            # Extract task details using NLP
            task_name = "New Task"  # Default name
            task_description = input_text
            task_action = "learn_information"  # Default action
            task_params = {"text": input_text}
            
            # Try to extract a better name from the input
            name_patterns = [
                r"task (?:called|named) ['\"]?([^'\"]+)['\"]?",
                r"create (?:a|the) ['\"]?([^'\"]+)['\"]? task"
            ]
            
            import re
            for pattern in name_patterns:
                match = re.search(pattern, input_text, re.IGNORECASE)
                if match:
                    task_name = match.group(1).strip()
                    break
            
            # Create the task
            task_result = self.create_task(
                name=task_name,
                description=task_description,
                action=task_action,
                params=task_params
            )
            
            return f"I've created a new task: '{task_name}' with ID {task_result['task_id']}. I'll work on it in the background."
            
        elif "task status" in input_text.lower() or "check task" in input_text.lower():
            # Extract task ID from input
            import re
            match = re.search(r"task (?:id )?['\"]?([a-f0-9-]+)['\"]?", input_text, re.IGNORECASE)
            
            if match:
                task_id = match.group(1).strip()
                status = self.get_task_status(task_id)
                
                if "error" in status:
                    return f"I couldn't find that task. {status['error']}"
                    
                return f"Task '{status['name']}' is {status['status']}. " + \
                       (f"It's {int(status['progress'] * 100)}% complete. " if status['progress'] > 0 else "") + \
                       (f"Result: {status['result']}" if status['result'] else "")
            else:
                # Show all tasks
                tasks = self.get_all_tasks()
                
                if tasks["total"] == 0:
                    return "You don't have any tasks yet. You can create one by saying 'create a task'."
                    
                return f"You have {tasks['total']} tasks: " + \
                       f"{tasks['pending']} pending, {tasks['in_progress']} in progress, " + \
                       f"{tasks['completed']} completed, and {tasks['failed']} failed."
                       
        elif "cancel task" in input_text.lower() or "delete task" in input_text.lower():
            # Extract task ID from input
            import re
            match = re.search(r"task (?:id )?['\"]?([a-f0-9-]+)['\"]?", input_text, re.IGNORECASE)
            
            if match:
                task_id = match.group(1).strip()
                result = self.cancel_task(task_id)
                
                if "error" in result:
                    return f"I couldn't cancel that task. {result['error']}"
                    
                return f"I've cancelled the task with ID {task_id}."
            else:
                return "Please specify which task you want to cancel by including the task ID."
                
        elif "learn" in input_text.lower() and len(input_text) > 10:
            # Learn new information
            result = self.learn_information(input_text)
            
            if result["success"]:
                return f"I've learned something new! {result['message']}"
            else:
                return f"I couldn't learn from that input. {result['message']}"
                
        elif "define" in input_text.lower() or "what is" in input_text.lower() or "what are" in input_text.lower():
            # Extract the term to define
            import re
            patterns = [
                r"define ['\"]?([^'\"?]+)['\"]?",
                r"what (?:is|are) ['\"]?([^'\"?]+)['\"]?",
                r"meaning of ['\"]?([^'\"?]+)['\"]?"
            ]
            
            for pattern in patterns:
                match = re.search(pattern, input_text, re.IGNORECASE)
                if match:
                    term = match.group(1).strip()
                    definition = self.get_definition(term)
                    
                    if "error" in definition:
                        # If no definition found, create a task to learn it
                        self.create_task(
                            name=f"Learn about {term}",
                            description=f"Research and learn about {term}",
                            action="learn_information",
                            params={"text": f"Research about {term}"}
                        )
                        
                        return f"I don't have a definition for '{term}' yet, but I've created a task to learn about it."
                        
                    return f"{term}: {definition['definition']}"
                    
            # If no term found, use the default processing
            pass
        
        # Create context for decision-making
        context = {
            "input": input_text,
            "user_name": user_name,
            "timestamp": time.time(),
            "nlp_analysis": nlp_analysis
        }
        
        # Store in short-term memory
        self.memory.store_short_term("last_input", context)
        
        # Make a decision
        decision = self.decision_engine.make_decision(context)
        
        # Execute the decided action
        action_name = decision.get("action", "default_response")
        params = decision.get("params", {})
        action_result = self.action_executor.execute_action(action_name, params)
        
        # Record the experience for learning
        self.learning_system.record_experience(context, decision, action_result)
        
        # Update conversation history
        history = self.memory.retrieve_long_term("conversation_history") or []
        history.append({
            "input": input_text,
            "user_name": user_name,
            "action": action_name,
            "params": params,
            "response": action_result.get("result", ""),
            "timestamp": time.time()
        })
        self.memory.store_long_term("conversation_history", history)
        
        # Store user preferences based on the interaction
        user_id = user_name.lower().replace(" ", "_")
        
        # If the user asked about a specific topic, store it as an interest
        for intent in nlp_analysis.get("intents", []):
            if intent in ["weather", "time", "joke", "fact"]:
                self.knowledge_base.add_user_preference(user_id, f"interest_{intent}", True)
                
        # Store sentiment
        sentiment = nlp_analysis.get("sentiment", {}).get("label")
        if sentiment:
            self.knowledge_base.add_user_preference(user_id, "last_sentiment", sentiment)
            
        # Return the result to the user
        if action_result.get("status") == "success":
            return action_result.get("result", "I processed your request.")
        else:
            return f"I encountered an issue: {action_result.get('message', 'Unknown error')}"
            
    def save_state(self, path: str = "agent_state.json") -> None:
        """Save the agent's state to disk."""
        self.memory.save_to_disk(path)
        logger.info(f"Agent state saved to {path}")
        
    def load_state(self, path: str = "agent_state.json") -> None:
        """Load the agent's state from disk."""
        self.memory.load_from_disk(path)
        logger.info(f"Agent state loaded from {path}")
        
    def learn(self) -> None:
        """Trigger the learning process."""
        self.learning_system.learn_from_experiences()


if __name__ == "__main__":
    # Example usage
    agent = AutonomousAgent(name="SuperAI")
    
    # Process some inputs
    print(agent.process_input("Hello there!"))
    print(agent.process_input("What's the weather like?"))
    print(agent.process_input("What time is it?"))
    
    # Learn from experiences
    agent.learn()
    
    # Save state
    agent.save_state()