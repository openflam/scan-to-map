# Search Server

A Flask server that provides a search API for the semantic 3D search demo.

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Server

```bash
python app.py
```

The server will start on `http://localhost:5000`

## API Endpoints

### POST/GET /search
Returns a bounding box for the search query.

**Parameters:**
- `query` (string): The search query

**Response:**
```json
{
  "x_min": -0.1641819722126625,
  "y_min": -0.20034926138025555,
  "z_min": 0.2588292556589842,
  "x_max": -0.6151471354052642,
  "y_max": 0.1119962775637272,
  "z_max": 0.6770841372640612
}
```

**Example:**
```bash
# GET request
curl "http://localhost:5000/search?query=printer"

# POST request
curl -X POST http://localhost:5000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "printer"}'
```

### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "ok"
}
```

## Testing

For testing purposes, the `/search` endpoint currently returns the bounding box of the 3rd component from `/outputs/PrintersNoNeg/bbox_corners.json`, regardless of the query.

## CORS

CORS is enabled for all routes to allow the frontend to make requests from a different origin.
