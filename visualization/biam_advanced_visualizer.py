"""
BIAM Advanced Visualizer
Advanced visualization features for BIAM model
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch
from typing import Dict, Any, List, Tuple, Optional
import os
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import networkx as nx

class BIAMAdvancedVisualizer:
    """
    Advanced visualization class for BIAM model
    """
    
    def __init__(self, config):
        """
        Initialize advanced visualizer
        
        Args:
            config: BIAM configuration
        """
        self.config = config
        self.viz_dir = "advanced_visualizations"
        os.makedirs(self.viz_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def create_interactive_dashboard(self, model, data, targets, save_path=None):
        """
        Create interactive dashboard with multiple visualizations
        
        Args:
            model: BIAM model
            data: Input data
            targets: Target labels
            save_path: Path to save dashboard
            
        Returns:
            Path to saved dashboard
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Feature Importance', 'Model Predictions', 
                          'Missing Value Analysis', 'Sample Weights'),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "heatmap"}, {"type": "histogram"}]]
        )
        
        # Feature importance
        if hasattr(model, 'get_feature_importance'):
            importance = model.get_feature_importance()
            fig.add_trace(
                go.Bar(x=list(range(len(importance))), y=importance, name="Feature Importance"),
                row=1, col=1
            )
        
        # Model predictions
        with torch.no_grad():
            predictions = model(torch.tensor(data, dtype=torch.float32))
            if self.config.task == 'classification':
                predictions = torch.softmax(predictions, dim=1)
                predictions = predictions[:, 1].numpy()
            else:
                predictions = predictions.numpy().flatten()
        
        fig.add_trace(
            go.Scatter(x=targets, y=predictions, mode='markers', name="Predictions"),
            row=1, col=2
        )
        
        # Missing value analysis
        missing_data = np.isnan(data)
        fig.add_trace(
            go.Heatmap(z=missing_data.astype(int), colorscale='Reds', name="Missing Values"),
            row=2, col=1
        )
        
        # Sample weights (if available)
        if hasattr(model, 'weighting_network'):
            # Generate sample weights
            with torch.no_grad():
                test_losses = torch.ones(len(data), 1)
                weights = model.weighting_network(test_losses).numpy().flatten()
            
            fig.add_trace(
                go.Histogram(x=weights, name="Sample Weights"),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title="BIAM Model Interactive Dashboard",
            showlegend=False,
            height=800
        )
        
        if save_path is None:
            save_path = os.path.join(self.viz_dir, 'interactive_dashboard.html')
        
        fig.write_html(save_path)
        return save_path
    
    def create_3d_feature_visualization(self, data, targets, feature_indices=None, save_path=None):
        """
        Create 3D feature visualization
        
        Args:
            data: Input data
            targets: Target labels
            feature_indices: Indices of features to visualize
            save_path: Path to save visualization
            
        Returns:
            Path to saved visualization
        """
        if feature_indices is None:
            feature_indices = [0, 1, 2]
        
        # Select features
        X_3d = data[:, feature_indices]
        
        # Create 3D scatter plot
        fig = go.Figure(data=[go.Scatter3d(
            x=X_3d[:, 0],
            y=X_3d[:, 1],
            z=X_3d[:, 2],
            mode='markers',
            marker=dict(
                size=5,
                color=targets,
                colorscale='Viridis',
                opacity=0.8
            ),
            text=[f'Target: {t}' for t in targets],
            hovertemplate='Feature 1: %{x}<br>Feature 2: %{y}<br>Feature 3: %{z}<br>%{text}<extra></extra>'
        )])
        
        fig.update_layout(
            title='3D Feature Visualization',
            scene=dict(
                xaxis_title=f'Feature {feature_indices[0]}',
                yaxis_title=f'Feature {feature_indices[1]}',
                zaxis_title=f'Feature {feature_indices[2]}'
            ),
            width=800,
            height=600
        )
        
        if save_path is None:
            save_path = os.path.join(self.viz_dir, '3d_feature_visualization.html')
        
        fig.write_html(save_path)
        return save_path
    
    def create_tsne_visualization(self, data, targets, perplexity=30, save_path=None):
        """
        Create t-SNE visualization of data
        
        Args:
            data: Input data
            targets: Target labels
            perplexity: t-SNE perplexity parameter
            save_path: Path to save visualization
            
        Returns:
            Path to saved visualization
        """
        # Handle missing values
        data_clean = np.nan_to_num(data, nan=0.0)
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        X_tsne = tsne.fit_transform(data_clean)
        
        # Create scatter plot
        fig = go.Figure()
        
        # Plot each class separately
        unique_targets = np.unique(targets)
        colors = px.colors.qualitative.Set1
        
        for i, target in enumerate(unique_targets):
            mask = targets == target
            fig.add_trace(go.Scatter(
                x=X_tsne[mask, 0],
                y=X_tsne[mask, 1],
                mode='markers',
                name=f'Class {target}',
                marker=dict(color=colors[i % len(colors)], size=8),
                hovertemplate=f'Class: {target}<br>t-SNE 1: %{{x}}<br>t-SNE 2: %{{y}}<extra></extra>'
            ))
        
        fig.update_layout(
            title='t-SNE Visualization of Data',
            xaxis_title='t-SNE Component 1',
            yaxis_title='t-SNE Component 2',
            width=800,
            height=600
        )
        
        if save_path is None:
            save_path = os.path.join(self.viz_dir, 'tsne_visualization.html')
        
        fig.write_html(save_path)
        return save_path
    
    def create_pca_visualization(self, data, targets, n_components=2, save_path=None):
        """
        Create PCA visualization of data
        
        Args:
            data: Input data
            targets: Target labels
            n_components: Number of PCA components
            save_path: Path to save visualization
            
        Returns:
            Path to saved visualization
        """
        # Handle missing values
        data_clean = np.nan_to_num(data, nan=0.0)
        
        # Apply PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(data_clean)
        
        if n_components == 2:
            # 2D visualization
            fig = go.Figure()
            
            unique_targets = np.unique(targets)
            colors = px.colors.qualitative.Set1
            
            for i, target in enumerate(unique_targets):
                mask = targets == target
                fig.add_trace(go.Scatter(
                    x=X_pca[mask, 0],
                    y=X_pca[mask, 1],
                    mode='markers',
                    name=f'Class {target}',
                    marker=dict(color=colors[i % len(colors)], size=8),
                    hovertemplate=f'Class: {target}<br>PC1: %{{x}}<br>PC2: %{{y}}<extra></extra>'
                ))
            
            fig.update_layout(
                title='PCA Visualization of Data',
                xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)',
                yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)',
                width=800,
                height=600
            )
        
        else:
            # 3D visualization
            fig = go.Figure(data=[go.Scatter3d(
                x=X_pca[:, 0],
                y=X_pca[:, 1],
                z=X_pca[:, 2],
                mode='markers',
                marker=dict(
                    size=5,
                    color=targets,
                    colorscale='Viridis',
                    opacity=0.8
                ),
                text=[f'Target: {t}' for t in targets],
                hovertemplate='PC1: %{x}<br>PC2: %{y}<br>PC3: %{z}<br>%{text}<extra></extra>'
            )])
            
            fig.update_layout(
                title='3D PCA Visualization of Data',
                scene=dict(
                    xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)',
                    yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)',
                    zaxis_title=f'PC3 ({pca.explained_variance_ratio_[2]:.2%} variance)'
                ),
                width=800,
                height=600
            )
        
        if save_path is None:
            save_path = os.path.join(self.viz_dir, f'pca_{n_components}d_visualization.html')
        
        fig.write_html(save_path)
        return save_path
    
    def create_feature_interaction_network(self, model, feature_names=None, save_path=None):
        """
        Create feature interaction network visualization
        
        Args:
            model: BIAM model
            feature_names: Names of features
            save_path: Path to save visualization
            
        Returns:
            Path to saved visualization
        """
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(10)]
        
        # Get interaction weights from model
        if hasattr(model, 'get_interaction_weights'):
            interaction_weights = model.get_interaction_weights()
        else:
            # Generate interaction weights for visualization
            n_features = len(feature_names)
            interaction_weights = np.random.randn(n_features, n_features)
        
        # Create network graph
        G = nx.Graph()
        
        # Add nodes
        for i, name in enumerate(feature_names):
            G.add_node(i, name=name)
        
        # Add edges based on interaction weights
        threshold = np.percentile(np.abs(interaction_weights), 90)  # Top 10% interactions
        
        for i in range(len(feature_names)):
            for j in range(i + 1, len(feature_names)):
                weight = interaction_weights[i, j]
                if abs(weight) > threshold:
                    G.add_edge(i, j, weight=abs(weight))
        
        # Create plotly network visualization
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Extract node positions
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        
        # Create edge traces
        edge_x = []
        edge_y = []
        edge_info = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_info.append(f"{feature_names[edge[0]]} - {feature_names[edge[1]]}")
        
        # Create node trace
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=[feature_names[node] for node in G.nodes()],
            textposition="middle center",
            marker=dict(
                size=20,
                color='lightblue',
                line=dict(width=2, color='black')
            )
        )
        
        # Create edge trace
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='gray'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title='Feature Interaction Network',
                           titlefont_size=16,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[ dict(
                               text="Node size represents feature importance, edge thickness represents interaction strength",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002,
                               xanchor='left', yanchor='bottom',
                               font=dict(color="gray", size=12)
                           )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                       )
        
        if save_path is None:
            save_path = os.path.join(self.viz_dir, 'feature_interaction_network.html')
        
        fig.write_html(save_path)
        return save_path
    
    def create_training_animation(self, training_history, save_path=None):
        """
        Create animated training progress visualization
        
        Args:
            training_history: Training history dictionary
            save_path: Path to save animation
            
        Returns:
            Path to saved animation
        """
        epochs = training_history['epochs']
        train_loss = training_history['train_loss']
        val_loss = training_history['val_loss']
        
        # Create animated plot
        fig = go.Figure()
        
        # Add traces
        fig.add_trace(go.Scatter(
            x=epochs, y=train_loss,
            mode='lines',
            name='Training Loss',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=epochs, y=val_loss,
            mode='lines',
            name='Validation Loss',
            line=dict(color='red', width=2)
        ))
        
        # Add animation frames
        frames = []
        for i in range(0, len(epochs), max(1, len(epochs) // 50)):  # 50 frames max
            frame_data = [
                go.Scatter(x=epochs[:i+1], y=train_loss[:i+1], mode='lines', name='Training Loss'),
                go.Scatter(x=epochs[:i+1], y=val_loss[:i+1], mode='lines', name='Validation Loss')
            ]
            frames.append(go.Frame(data=frame_data, name=str(i)))
        
        fig.frames = frames
        
        # Add play button
        fig.update_layout(
            title='Training Progress Animation',
            xaxis_title='Epoch',
            yaxis_title='Loss',
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    {
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {'frame': {'duration': 100, 'redraw': True}}]
                    },
                    {
                        'label': 'Pause',
                        'method': 'animate',
                        'args': [[None], {'frame': {'duration': 0, 'redraw': False}}]
                    }
                ]
            }],
            width=800,
            height=600
        )
        
        if save_path is None:
            save_path = os.path.join(self.viz_dir, 'training_animation.html')
        
        fig.write_html(save_path)
        return save_path
    
    def create_heatmap_matrix(self, data, feature_names=None, save_path=None):
        """
        Create correlation heatmap matrix
        
        Args:
            data: Input data
            feature_names: Names of features
            save_path: Path to save heatmap
            
        Returns:
            Path to saved heatmap
        """
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(data.shape[1])]
        
        # Handle missing values
        data_clean = np.nan_to_num(data, nan=0.0)
        
        # Calculate correlation matrix
        df = pd.DataFrame(data_clean, columns=feature_names)
        corr_matrix = df.corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values,
            texttemplate="%{text:.2f}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Feature Correlation Matrix',
            width=800,
            height=600
        )
        
        if save_path is None:
            save_path = os.path.join(self.viz_dir, 'correlation_heatmap.html')
        
        fig.write_html(save_path)
        return save_path
