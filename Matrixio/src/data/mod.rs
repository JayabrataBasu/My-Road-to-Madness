pub mod project;
pub mod import_export;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use crate::matrix::Matrix;

/// Data layer for managing matrices and projects
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatrixCollection {
    pub matrices: HashMap<String, Matrix>,
    pub metadata: ProjectMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectMetadata {
    pub name: String,
    pub description: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub modified_at: chrono::DateTime<chrono::Utc>,
    pub version: String,
}

impl Default for ProjectMetadata {
    fn default() -> Self {
        let now = chrono::Utc::now();
        Self {
            name: "Untitled Project".to_string(),
            description: String::new(),
            created_at: now,
            modified_at: now,
            version: "1.0".to_string(),
        }
    }
}