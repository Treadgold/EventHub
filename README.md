# EventHub

A modern event management platform with AI-powered event creation.

## Features

- AI-powered event creation assistant using LangGraph and Ollama (GPT-oss 20b model)
- Role-based authentication (Admin, Event Organiser, User)
- User authentication and registration
- MongoDB database for persistent storage
- Chronological event listings
- Support for both online and in-person events
- Admin panel for user management

## User Roles

| Role | View Events | Buy Tickets | Create Events | Edit Own Events | Delete Own Events | Manage All Events | Manage Users |
|------|:-----------:|:-----------:|:-------------:|:---------------:|:-----------------:|:-----------------:|:------------:|
| **Not Logged In** | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| **User** | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| **Event Organiser** | ✓ | ✓ | ✓ | ✓ | Only if not live* | ✗ | ✗ |
| **Admin** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

*Event Organisers cannot delete their own events once they are "live" (start time has passed), as users may have already booked tickets.

## Quick Start with Docker

> **Prerequisites:** Ensure Ollama is running on your host machine with the GPT-oss 20b model installed. See [Ollama Setup](#ollama-setup) below.

1. **Start the application:**
   ```bash
   docker compose up
   ```

2. **Access the application:**
   - Open your browser to `http://localhost:8011`
   - The application will be available on `0.0.0.0:8011`

3. **Default Login Credentials:**

   The following test users are automatically created on startup:

   | Role | Username | Password | Permissions |
   |------|----------|----------|-------------|
   | **Admin** | `admin` | `admin123` | Full access: create/edit/delete events, manage users |
   | **Admin** | `admin2` | `admin123` | Full access: create/edit/delete events, manage users |
   | **Event Organiser** | `organiser1` | `organiser123` | Create events, edit own events, delete own events (only if not live) |
   | **Event Organiser** | `organiser2` | `organiser123` | Create events, edit own events, delete own events (only if not live) |
   | **User** | `user1` | `user123` | View events, buy tickets |
   | **User** | `user2` | `user123` | View events, buy tickets |

4. **Stop the application:**
   ```bash
   docker compose down
   ```

5. **Stop and remove volumes (clean slate):**
   ```bash
   docker compose down -v
   ```

## Services

- **Web Application**: FastAPI server running on port 8011
- **MongoDB**: Database server running on port 27017

## Environment Variables

The following environment variables can be customized in `docker-compose.yml`:

- `MONGODB_URL`: MongoDB connection string (default: `mongodb://mongodb:27017`)
- `DATABASE_NAME`: Database name (default: `event_website`)
- `SECRET_KEY`: Secret key for session management (change in production!)

## Development

For local development without Docker:

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Start MongoDB locally or use a cloud instance

3. Set environment variables:
   ```bash
   export MONGODB_URL="mongodb://localhost:27017"
   export DATABASE_NAME="event_website"
   export SECRET_KEY="your-secret-key"
   ```

4. Run the application:
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8011
   ```

## Requirements

- Docker and Docker Compose
- Python 3.11+ (for local development)
- MongoDB (handled by Docker Compose)
- **Ollama** running on the host machine with the **GPT-oss 20b** model

### Ollama Setup

The AI-powered event creation feature requires Ollama to be running on the host machine (the same machine running Docker). The application uses the GPT-oss 20b model.

1. **Install Ollama** from [ollama.com](https://ollama.com)

2. **Pull the GPT-oss 20b model:**
   ```bash
   ollama pull gpt-oss:20b
   ```

3. **Ensure Ollama is running** before starting the Docker containers:
   ```bash
   ollama serve
   ```

> **Note:** The GPT-oss 20b model requires significant system resources. Ensure your machine has adequate RAM and GPU capabilities to run this model.

> **Windows + WSL Users:** If running Docker in WSL while Ollama runs on Windows, you may need to configure the Ollama URL to point to your Windows host IP (e.g., `192.168.137.146:11434`) instead of `localhost`.

