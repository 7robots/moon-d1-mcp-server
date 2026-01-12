"""
MCP Server for Cloudflare D1 Selenography Database.

This server provides tools to interact with a collection of lunar surface features
(craters, maria, mountains, etc.) stored in a Cloudflare D1 database.
"""

import json
import os
from enum import Enum
from typing import Any, Dict, List, Optional

import httpx
from fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict, Field, field_validator

# Constants (configurable via environment variables)
CLOUDFLARE_ID = os.environ.get("CLOUDFLARE_ID", "")
CLOUDFLARE_TOKEN = os.environ.get("CLOUDFLARE_TOKEN", "")
DATABASE_ID = os.environ.get("DATABASE_ID", "")

# D1 API endpoint
D1_API_URL = f"https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ID}/d1/database/{DATABASE_ID}/query"

# Initialize MCP server
mcp = FastMCP(
    "moon-d1-mcp",
    instructions="MCP server for exploring lunar selenography data. Use these tools to search, filter, and analyze features on the Moon's surface including craters, maria, mountains, and more.",
)


# Enums
class ResponseFormat(str, Enum):
    """Output format for tool responses."""

    MARKDOWN = "markdown"
    JSON = "json"


# Pydantic Input Models
class ListFeaturesInput(BaseModel):
    """Input model for listing lunar features."""

    model_config = ConfigDict(str_strip_whitespace=True)

    limit: int = Field(
        default=20,
        description="Maximum number of features to return",
        ge=1,
        le=100,
    )
    offset: int = Field(
        default=0, description="Number of features to skip for pagination", ge=0
    )
    feature_type: Optional[str] = Field(
        default=None,
        description="Filter by feature type (e.g., 'Crater, craters', 'Mare, maria', 'Mons, montes')",
    )
    min_diameter: Optional[float] = Field(
        default=None, description="Minimum diameter in km", ge=0
    )
    max_diameter: Optional[float] = Field(
        default=None, description="Maximum diameter in km", ge=0
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.JSON,
        description="Output format: 'json' for structured data or 'markdown' for human-readable",
    )


class GetFeatureInput(BaseModel):
    """Input model for getting a single feature."""

    model_config = ConfigDict(str_strip_whitespace=True)

    feature_id: int = Field(
        ..., description="The Feature_ID of the lunar feature"
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.JSON,
        description="Output format: 'json' for structured data or 'markdown' for human-readable",
    )


class SearchFeaturesInput(BaseModel):
    """Input model for searching features by name."""

    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(
        ...,
        description="Search string to match against feature names",
        min_length=1,
        max_length=100,
    )
    limit: int = Field(default=20, description="Maximum results to return", ge=1, le=100)
    response_format: ResponseFormat = Field(
        default=ResponseFormat.JSON,
        description="Output format: 'json' for structured data or 'markdown' for human-readable",
    )


class NearbyFeaturesInput(BaseModel):
    """Input model for finding features near coordinates."""

    model_config = ConfigDict(str_strip_whitespace=True)

    longitude: float = Field(
        ..., description="Center longitude (-180 to 180)", ge=-180, le=180
    )
    latitude: float = Field(
        ..., description="Center latitude (-90 to 90)", ge=-90, le=90
    )
    radius: float = Field(
        default=10.0, description="Search radius in degrees", ge=0.1, le=90
    )
    limit: int = Field(default=20, description="Maximum results to return", ge=1, le=100)
    response_format: ResponseFormat = Field(
        default=ResponseFormat.JSON,
        description="Output format: 'json' for structured data or 'markdown' for human-readable",
    )


class GetStatsInput(BaseModel):
    """Input model for getting collection statistics."""

    response_format: ResponseFormat = Field(
        default=ResponseFormat.JSON,
        description="Output format: 'json' for structured data or 'markdown' for human-readable",
    )


class GetTypesInput(BaseModel):
    """Input model for getting feature types."""

    response_format: ResponseFormat = Field(
        default=ResponseFormat.JSON,
        description="Output format: 'json' for structured data or 'markdown' for human-readable",
    )


# Utility functions
async def execute_d1_query(sql: str, params: List[Any] = None) -> Dict[str, Any]:
    """Execute a SQL query against the D1 database."""
    headers = {
        "Authorization": f"Bearer {CLOUDFLARE_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {"sql": sql}
    if params:
        payload["params"] = params

    async with httpx.AsyncClient() as client:
        response = await client.post(
            D1_API_URL,
            headers=headers,
            json=payload,
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json()


def format_feature_markdown(feature: Dict[str, Any]) -> str:
    """Format a lunar feature as markdown."""
    lines = [f"## {feature.get('Name', 'Unknown')} (ID: {feature.get('Feature_ID', 'N/A')})"]
    lines.append(f"**Type**: {feature.get('Type', 'Unknown')}")

    diameter = feature.get('Diameter')
    if diameter:
        lines.append(f"**Diameter**: {diameter:.2f} km")

    lon = feature.get('Center_Lon')
    lat = feature.get('Center_Lat')
    if lon is not None and lat is not None:
        lines.append(f"**Coordinates**: {lat:.2f}° lat, {lon:.2f}° lon")

    desc = feature.get('Description')
    if desc:
        lines.append(f"**Description**: {desc}")

    usgs = feature.get('USGS')
    if usgs:
        lines.append(f"**USGS**: {usgs}")

    wiki = feature.get('Wikipedia')
    if wiki:
        lines.append(f"**Wikipedia**: {wiki}")

    return "\n".join(lines)


def handle_api_error(e: Exception) -> str:
    """Format API errors consistently."""
    if isinstance(e, httpx.HTTPStatusError):
        if e.response.status_code == 401:
            return "Error: Authentication failed. Check CLOUDFLARE_TOKEN."
        elif e.response.status_code == 404:
            return "Error: Database not found. Check DATABASE_ID."
        return f"Error: API request failed with status {e.response.status_code}"
    elif isinstance(e, httpx.TimeoutException):
        return "Error: Request timed out. Please try again."
    return f"Error: {type(e).__name__}: {e}"


# Tool definitions
@mcp.tool()
async def moon_list_features(
    limit: int = 20,
    offset: int = 0,
    feature_type: Optional[str] = None,
    min_diameter: Optional[float] = None,
    max_diameter: Optional[float] = None,
    response_format: str = "json",
) -> str:
    """List lunar surface features with optional filtering and pagination.

    Args:
        limit: Max features to return (1-100, default 20)
        offset: Features to skip for pagination (default 0)
        feature_type: Filter by type (e.g., 'Crater, craters', 'Mare, maria')
        min_diameter: Minimum diameter in km
        max_diameter: Maximum diameter in km
        response_format: 'json' for structured data or 'markdown' for human-readable
    """
    try:
        limit = max(1, min(100, limit))
        offset = max(0, offset)

        conditions = []
        params = []

        if feature_type:
            conditions.append("Type LIKE ?")
            params.append(f"%{feature_type}%")
        if min_diameter is not None:
            conditions.append("Diameter >= ?")
            params.append(min_diameter)
        if max_diameter is not None:
            conditions.append("Diameter <= ?")
            params.append(max_diameter)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        # Get total count
        count_sql = f"SELECT COUNT(*) as total FROM selenography {where_clause}"
        count_result = await execute_d1_query(count_sql, params if params else None)
        total = count_result["result"][0]["results"][0]["total"]

        # Get features
        sql = f"""
            SELECT Feature_ID, Name, Description, Diameter, Center_Lon, Center_Lat, Type, USGS, Wikipedia
            FROM selenography
            {where_clause}
            ORDER BY Diameter DESC
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])

        result = await execute_d1_query(sql, params)
        features = result["result"][0]["results"]

        if response_format == "markdown":
            if not features:
                return "No lunar features found matching the criteria."

            lines = ["# Lunar Features", ""]
            lines.append(f"Showing {len(features)} of {total} features (offset: {offset})")
            lines.append("")

            for feature in features:
                lines.append(format_feature_markdown(feature))
                lines.append("")

            if total > offset + len(features):
                lines.append(f"*{total - offset - len(features)} more features available. Use offset={offset + len(features)} to see next page.*")

            return "\n".join(lines)

        return json.dumps(
            {
                "total": total,
                "count": len(features),
                "offset": offset,
                "has_more": total > offset + len(features),
                "next_offset": offset + len(features) if total > offset + len(features) else None,
                "features": features,
            },
            indent=2,
        )
    except Exception as e:
        return handle_api_error(e)


@mcp.tool()
async def moon_get_feature(feature_id: int, response_format: str = "json") -> str:
    """Get a specific lunar feature by its Feature_ID.

    Args:
        feature_id: The numeric Feature_ID of the lunar feature
        response_format: 'json' for structured data or 'markdown' for human-readable
    """
    try:
        sql = """
            SELECT Feature_ID, Name, Description, Diameter, Center_Lon, Center_Lat, Type, USGS, Wikipedia
            FROM selenography
            WHERE Feature_ID = ?
        """
        result = await execute_d1_query(sql, [feature_id])
        features = result["result"][0]["results"]

        if not features:
            return f"Error: Feature not found with ID {feature_id}. Use moon_list_features or moon_search_features to find valid IDs."

        feature = features[0]

        if response_format == "markdown":
            return format_feature_markdown(feature)

        return json.dumps(feature, indent=2)
    except Exception as e:
        return handle_api_error(e)


@mcp.tool()
async def moon_search_features(query: str, limit: int = 20, response_format: str = "json") -> str:
    """Search for lunar features by name.

    Args:
        query: Search string to match against feature names (case-insensitive)
        limit: Maximum results (1-100, default 20)
        response_format: 'json' for structured data or 'markdown' for human-readable
    """
    try:
        limit = max(1, min(100, limit))

        sql = """
            SELECT Feature_ID, Name, Description, Diameter, Center_Lon, Center_Lat, Type, USGS, Wikipedia
            FROM selenography
            WHERE Name LIKE ?
            ORDER BY Diameter DESC
            LIMIT ?
        """
        result = await execute_d1_query(sql, [f"%{query}%", limit])
        features = result["result"][0]["results"]

        if not features:
            return f"No lunar features found matching '{query}'."

        if response_format == "markdown":
            lines = [f"# Search Results: '{query}'", ""]
            lines.append(f"Found {len(features)} feature(s)")
            lines.append("")

            for feature in features:
                lines.append(format_feature_markdown(feature))
                lines.append("")

            return "\n".join(lines)

        return json.dumps({"count": len(features), "features": features}, indent=2)
    except Exception as e:
        return handle_api_error(e)


@mcp.tool()
async def moon_features_near(
    longitude: float,
    latitude: float,
    radius: float = 10.0,
    limit: int = 20,
    response_format: str = "json",
) -> str:
    """Find lunar features near specific coordinates.

    Args:
        longitude: Center longitude (-180 to 180)
        latitude: Center latitude (-90 to 90)
        radius: Search radius in degrees (default 10.0)
        limit: Maximum results (1-100, default 20)
        response_format: 'json' for structured data or 'markdown' for human-readable
    """
    try:
        limit = max(1, min(100, limit))

        # Simple bounding box query (approximation)
        sql = """
            SELECT Feature_ID, Name, Description, Diameter, Center_Lon, Center_Lat, Type, USGS, Wikipedia,
                   ABS(Center_Lon - ?) + ABS(Center_Lat - ?) as distance
            FROM selenography
            WHERE Center_Lon BETWEEN ? AND ?
              AND Center_Lat BETWEEN ? AND ?
            ORDER BY distance ASC
            LIMIT ?
        """
        params = [
            longitude, latitude,
            longitude - radius, longitude + radius,
            latitude - radius, latitude + radius,
            limit
        ]

        result = await execute_d1_query(sql, params)
        features = result["result"][0]["results"]

        if not features:
            return f"No lunar features found within {radius}° of ({latitude}°, {longitude}°)."

        if response_format == "markdown":
            lines = [f"# Features Near ({latitude:.2f}°, {longitude:.2f}°)", ""]
            lines.append(f"Found {len(features)} feature(s) within {radius}° radius")
            lines.append("")

            for feature in features:
                dist = feature.pop('distance', None)
                lines.append(format_feature_markdown(feature))
                if dist is not None:
                    lines.append(f"*Distance: ~{dist:.2f}°*")
                lines.append("")

            return "\n".join(lines)

        return json.dumps({"count": len(features), "center": {"longitude": longitude, "latitude": latitude}, "radius": radius, "features": features}, indent=2)
    except Exception as e:
        return handle_api_error(e)


@mcp.tool()
async def moon_get_feature_types(response_format: str = "json") -> str:
    """Get all unique feature types in the database with counts.

    Args:
        response_format: 'json' for structured data or 'markdown' for human-readable
    """
    try:
        sql = """
            SELECT Type, COUNT(*) as count
            FROM selenography
            GROUP BY Type
            ORDER BY count DESC
        """
        result = await execute_d1_query(sql)
        types = result["result"][0]["results"]

        if response_format == "markdown":
            lines = ["# Lunar Feature Types", ""]
            lines.append(f"Total: {len(types)} unique types")
            lines.append("")

            for t in types:
                lines.append(f"- **{t['Type']}**: {t['count']} features")

            return "\n".join(lines)

        return json.dumps({"count": len(types), "types": types}, indent=2)
    except Exception as e:
        return handle_api_error(e)


@mcp.tool()
async def moon_get_stats(response_format: str = "json") -> str:
    """Get aggregate statistics about the lunar feature collection.

    Args:
        response_format: 'json' for structured data or 'markdown' for human-readable
    """
    try:
        # Total count
        count_result = await execute_d1_query("SELECT COUNT(*) as total FROM selenography")
        total = count_result["result"][0]["results"][0]["total"]

        # Diameter stats
        diameter_sql = """
            SELECT
                AVG(Diameter) as avg_diameter,
                MIN(Diameter) as min_diameter,
                MAX(Diameter) as max_diameter
            FROM selenography
            WHERE Diameter IS NOT NULL AND Diameter > 0
        """
        diameter_result = await execute_d1_query(diameter_sql)
        diameter_stats = diameter_result["result"][0]["results"][0]

        # Type distribution (top 10)
        type_sql = """
            SELECT Type, COUNT(*) as count
            FROM selenography
            GROUP BY Type
            ORDER BY count DESC
            LIMIT 10
        """
        type_result = await execute_d1_query(type_sql)
        type_dist = {t["Type"]: t["count"] for t in type_result["result"][0]["results"]}

        # Largest features
        largest_sql = """
            SELECT Name, Type, Diameter
            FROM selenography
            WHERE Diameter IS NOT NULL
            ORDER BY Diameter DESC
            LIMIT 5
        """
        largest_result = await execute_d1_query(largest_sql)
        largest = largest_result["result"][0]["results"]

        stats = {
            "total_features": total,
            "diameter_stats": {
                "average_km": round(diameter_stats["avg_diameter"], 2) if diameter_stats["avg_diameter"] else None,
                "minimum_km": diameter_stats["min_diameter"],
                "maximum_km": diameter_stats["max_diameter"],
            },
            "type_distribution": type_dist,
            "largest_features": largest,
        }

        if response_format == "markdown":
            lines = ["# Lunar Selenography Statistics", ""]
            lines.append(f"**Total Features**: {total}")
            lines.append("")

            lines.append("## Diameter Statistics")
            lines.append(f"- Average: {stats['diameter_stats']['average_km']:.2f} km")
            lines.append(f"- Smallest: {stats['diameter_stats']['minimum_km']:.2f} km")
            lines.append(f"- Largest: {stats['diameter_stats']['maximum_km']:.2f} km")
            lines.append("")

            lines.append("## Feature Type Distribution (Top 10)")
            for type_name, count in type_dist.items():
                lines.append(f"- {type_name}: {count}")
            lines.append("")

            lines.append("## Largest Features")
            for f in largest:
                lines.append(f"- {f['Name']} ({f['Type']}): {f['Diameter']:.2f} km")

            return "\n".join(lines)

        return json.dumps(stats, indent=2)
    except Exception as e:
        return handle_api_error(e)


if __name__ == "__main__":
    # HTTP transport for fastmcp.cloud
    host = os.getenv("MCP_HOST", "0.0.0.0")
    port = int(os.getenv("MCP_PORT", "8000"))
    mcp.run(transport="http", host=host, port=port)
