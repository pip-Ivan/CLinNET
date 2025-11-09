"""
Interactive Network Visualization Module for CLinNet

Single-file implementation with all functionality for interactive
network visualization with dynamic controls.
"""

import networkx as nx
import pandas as pd
import numpy as np
import json
from typing import List, Optional, Dict
from pyvis.network import Network
import warnings
warnings.filterwarnings('ignore')


def recunstruct_shap_network(shap_graph) -> nx.DiGraph:
    """
    Reconstruct NetworkX DiGraph from SHAP graph structure.
    
    Args:
        shap_graph: SHAP graph object with edges and nodes

    Returns:
        NetworkX DiGraph representation of the SHAP graph
    """
    G = nx.DiGraph()

    for pre_layer, post_layer in shap_graph.edges:
        cm = shap_graph.edges[(pre_layer, post_layer)]['cm']
        rows, cols = np.where(cm == 1)
        edges = [(cm.index[row], cm.columns[col]) for row, col in zip(rows, cols)]
        G.add_edges_from(edges)
        layer_attr = {k:v for k, v in shap_graph.nodes[pre_layer].items() if k in ['feature_id', 'feature_name', 'network', 'layer_id', 'rank_index']}
        layer_attr['layer_name'] = pre_layer
        node_attr = pd.DataFrame(layer_attr).set_index('feature_id').to_dict('index')
        nx.set_node_attributes(G, node_attr)

    return G


def _extract_subgraph_by_features(
    graph: nx.DiGraph,
    feature_ids: List[str], 
    include_neighbors: bool = True,
    neighbor_depth: int = 1,
    rank_cutoff: Optional[int] = None,
    show_trajectory: bool = False,
    root_node: str = 'root'
) -> nx.DiGraph:
    """
    Extract subgraph containing specified feature IDs.
    
    Args:
        graph: NetworkX DiGraph
        feature_ids: List of feature IDs to include
        include_neighbors: Whether to include neighboring nodes
        neighbor_depth: How many levels of neighbors to include
        rank_cutoff: Only include nodes with rank_index < cutoff
        show_trajectory: Include all nodes on paths from feature_ids to root_node
        root_node: Name of the root node for trajectory analysis
        
    Returns:
        NetworkX DiGraph containing the subgraph
    """
    if not feature_ids:
        raise ValueError("feature_ids cannot be empty")
        
    # Filter feature_ids to only include nodes that exist in the graph
    existing_features = [fid for fid in feature_ids if fid in graph.nodes()]
    if not existing_features:
        raise ValueError("None of the specified feature_ids exist in the graph")
        
    subgraph_nodes = set(existing_features)
    
    # Add trajectory nodes if requested
    trajectory_nodes = set()
    if show_trajectory and root_node in graph.nodes():
        for feature in existing_features:
            try:
                paths = list(nx.all_simple_paths(graph, feature, root_node, cutoff=10))
                for path in paths:
                    trajectory_nodes.update(path)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                try:
                    paths = list(nx.all_simple_paths(graph, root_node, feature, cutoff=10))
                    for path in paths:
                        trajectory_nodes.update(path)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
        
        subgraph_nodes.update(trajectory_nodes)
        print(f"Added {len(trajectory_nodes)} trajectory nodes from features to {root_node}")
    
    if include_neighbors:
        for _ in range(neighbor_depth):
            new_nodes = set()
            for node in subgraph_nodes:
                new_nodes.update(graph.predecessors(node))
                new_nodes.update(graph.successors(node))
            subgraph_nodes.update(new_nodes)
    
    # Apply rank filtering - ALWAYS preserve feature_ids, trajectory nodes, and root
    if rank_cutoff is not None:
        filtered_nodes = set()
        for node in subgraph_nodes:
            if node in graph.nodes():
                # Always include: feature_ids, trajectory nodes, and root node
                if node in existing_features or node in trajectory_nodes or node == root_node:
                    filtered_nodes.add(node)
                    continue
                
                # For other nodes, apply rank filtering
                rank = graph.nodes[node].get('rank_index', float('inf'))
                try:
                    if isinstance(rank, (int, float)) and rank < rank_cutoff:
                        filtered_nodes.add(node)
                    elif rank == 'N/A':
                        filtered_nodes.add(node)
                except (TypeError, ValueError):
                    filtered_nodes.add(node)
        subgraph_nodes = filtered_nodes
        
        if not subgraph_nodes:
            print(f"Warning: No nodes found with rank < {rank_cutoff}. Returning empty graph.")
        else:
            print(f"Rank filtering: Kept {len(existing_features)} target features + {len(trajectory_nodes)} trajectory nodes + {len(subgraph_nodes) - len(existing_features) - len(trajectory_nodes)} high-ranking nodes")
    
    # Create subgraph
    subgraph = graph.subgraph(subgraph_nodes).copy()
    return subgraph


def _serialize_graph_data(graph: nx.DiGraph) -> str:
    """Serialize full graph data for JavaScript access."""
    graph_data = {
        'nodes': {},
        'edges': []
    }
    
    # Serialize nodes
    for node in graph.nodes():
        attrs = graph.nodes[node]
        graph_data['nodes'][node] = {
            'feature_name': attrs.get('feature_name', node),
            'layer_name': attrs.get('layer_name', 'N/A'),
            'network': attrs.get('network', 'N/A'),
            'rank_index': attrs.get('rank_index', 999999),
            'layer_id': attrs.get('layer_id', 'N/A')
        }
    
    # Serialize edges
    for edge in graph.edges():
        graph_data['edges'].append({
            'from': edge[0],
            'to': edge[1]
        })
    
    return json.dumps(graph_data)


def create_interactive_network(
    graph: nx.DiGraph,
    feature_ids: List[str],
    include_neighbors: bool = True,
    neighbor_depth: int = 1,
    initial_rank_cutoff: Optional[int] = 100,
    show_trajectory: bool = False,
    root_node: str = 'root',
    filename: str = "network.html",
    auto_open: bool = False
) -> str:

    # Get initial subgraph
    subgraph = _extract_subgraph_by_features(
        graph=graph,
        feature_ids=feature_ids,
        include_neighbors=include_neighbors,
        neighbor_depth=neighbor_depth,
        rank_cutoff=initial_rank_cutoff,
        show_trajectory=show_trajectory,
        root_node=root_node
    )
    
    # Create Pyvis network
    width = "100%"
    height = "900px"
    net = Network(width=width, height=height, bgcolor="#ffffff", font_color="black")
    
    # Configure physics
    net.set_options("""
    var options = {
            "nodes": {
                "font": { "face": "Times New Roman, Times, serif" }
            },
      "physics": {
        "enabled": true,
        "stabilization": {"enabled": true, "iterations": 100},
        "barnesHut": {
          "gravitationalConstant": -8000,
          "centralGravity": 0.3,
          "springLength": 95,
          "springConstant": 0.04,
          "damping": 0.09,
          "avoidOverlap": 0.1
        }
      },
      "interaction": {
        "hover": true,
        "multiselect": true,
        "selectConnectedEdges": false
      }
    }
    """)
    
    # Color scheme
    colors = {
        'feature': '#FF6B6B',
        'root': '#9B59B6',
        'trajectory': '#F39C12',
        'neighbor': '#4ECDC4',
        'network_GO': '#45B7D1',
        'network_Reactome': '#96CEB4',
        'default': '#FECA57',
        'selected': '#FF1744'
    }
    
    # Find trajectory nodes
    trajectory_nodes = set()
    if show_trajectory and root_node in subgraph.nodes():
        for feature in feature_ids:
            if feature in subgraph.nodes():
                try:
                    paths = list(nx.all_simple_paths(subgraph, feature, root_node, cutoff=10))
                    for path in paths:
                        trajectory_nodes.update(path[1:-1])
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    try:
                        paths = list(nx.all_simple_paths(subgraph, root_node, feature, cutoff=10))
                        for path in paths:
                            trajectory_nodes.update(path[1:-1])
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        continue
    
    # Add nodes to network
    for node in subgraph.nodes():
        attrs = subgraph.nodes[node]
        
        # Determine node properties
        if node in feature_ids:
            color = colors['feature']
            size = 25
            node_type = "feature"
        elif node == root_node:
            color = colors['root']
            size = 30
            node_type = "root"
        elif node in trajectory_nodes:
            color = colors['trajectory']
            size = 20
            node_type = "trajectory"
        elif attrs.get('network') == 'GO':
            color = colors['network_GO']
            size = 20
            node_type = "GO"
        elif attrs.get('network') == 'Reactome':
            color = colors['network_Reactome']
            size = 20
            node_type = "Reactome"
        else:
            color = colors['neighbor']
            size = 15
            node_type = "neighbor"
        
        # Create hover info
        feature_name = attrs.get('feature_name', node)
        hover_info = []
        for key, value in attrs.items():
            if key in ['feature_id', 'feature_name', 'layer_name', 'network', 'rank_index']:
                hover_info.append(f"{key}: {value}")
        hover_info.append(f"In-degree: {subgraph.in_degree(node)}")
        hover_info.append(f"Out-degree: {subgraph.out_degree(node)}")
        hover_info.append("Double-click to expand neighbors")
        
        # Use feature_name for label
        display_label = feature_name if len(feature_name) <= 15 else feature_name[:12] + "..."
        
        # Add node
        net.add_node(
            node,
            label=display_label,
            title="\n".join(hover_info),
            color=color,
            size=size,
            font={'size': 12, 'face': 'Times New Roman, Times, serif'},
            type=node_type
        )
    
    # Add edges
    for edge in subgraph.edges():
        source, target = edge
        net.add_edge(
            source, target,
            color={'color': '#888888', 'width': 2},
            arrows={'to': {'enabled': True, 'scaleFactor': 0.8}}
        )
    
    # Save basic HTML
    net.save_graph(filename)
    
    # HTML with custom JavaScript controls
    _html_with_controls(
        filename=filename,
        full_graph_data=_serialize_graph_data(graph),
        initial_rank_cutoff=initial_rank_cutoff,
        feature_ids=feature_ids,
        colors=colors
    )
    
    print(f"✅ Interactive network saved as '{filename}'")
    
    if auto_open:
        import webbrowser
        import os
        abs_path = os.path.abspath(filename)
        webbrowser.open(f'file://{abs_path}')
        print(f"✅ Opening visualization in browser: {abs_path}")
    
    return filename


def _html_with_controls(
    filename: str,
    full_graph_data: str,
    initial_rank_cutoff: Optional[int],
    feature_ids: List[str],
    colors: Dict[str, str]
):
    """
     HTML file with custom JavaScript controls.
    """
    # Read the generated HTML
    with open(filename, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Create custom controls HTML
    controls_html = f"""
    <div id="controls-panel" style="position: absolute; top: 10px; left: 10px; 
         background: white; padding: 15px; border-radius: 8px; 
         box-shadow: 0 2px 10px rgba(0,0,0,0.1); z-index: 1000; max-width: 300px;">
        <h3 style="margin-top: 0; color: #333;">Network Controls</h3>
        
        <!-- Rank Cutoff Control -->
        <div style="margin-bottom: 15px;">
            <label style="font-weight: bold; color: #555;">Rank Cutoff:</label>
            <input type="range" id="rank-slider" min="10" max="1000" value="{initial_rank_cutoff or 100}" 
                   style="width: 100%;">
            <span id="rank-value" style="color: #666;">{initial_rank_cutoff or 100}</span>
            <button onclick="applyRankFilter()" 
                    style="background: #4CAF50; color: white; border: none; 
                           padding: 8px 15px; border-radius: 4px; cursor: pointer; margin-top: 5px;">
                Apply Filter
            </button>
        </div>
        
        <!-- Selection Info -->
        <div style="margin-bottom: 15px; padding: 10px; background: #f5f5f5; border-radius: 4px;">
            <strong style="color: #333;">Selected Nodes:</strong>
            <div id="selection-info" style="color: #666; margin-top: 5px;">None</div>
        </div>
        
        <!-- Action Buttons -->
        <div style="margin-bottom: 10px;">
            <button onclick="expandSelectedNodes()" 
                    style="background: #2196F3; color: white; border: none; 
                           padding: 10px 15px; border-radius: 4px; cursor: pointer; width: 100%; margin-bottom: 8px;">
                ➕ Expand 1st Degree
            </button>
            <button onclick="removeSelectedNodes()" 
                    style="background: #f44336; color: white; border: none; 
                           padding: 10px 15px; border-radius: 4px; cursor: pointer; width: 100%; margin-bottom: 8px;">
                🗑️ Remove Selected
            </button>
            <button onclick="removeFloatingNodes()" 
                    style="background: #E91E63; color: white; border: none; 
                           padding: 10px 15px; border-radius: 4px; cursor: pointer; width: 100%; margin-bottom: 8px;">
                🧹 Remove Floating Nodes
            </button>
            <button onclick="resetSelection()" 
                    style="background: #FF9800; color: white; border: none; 
                           padding: 10px 15px; border-radius: 4px; cursor: pointer; width: 100%; margin-bottom: 8px;">
                ↩️ Clear Selection
            </button>
            <button onclick="resetNetwork()" 
                    style="background: #9C27B0; color: white; border: none; 
                           padding: 10px 15px; border-radius: 4px; cursor: pointer; width: 100%;">
                🔄 Reset Network
            </button>
        </div>
        
        <!-- Legend -->
        <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #ddd;">
            <strong style="color: #333;">Legend:</strong>
            <div style="margin-top: 8px; font-size: 12px;">
                <div><span style="color: {colors['feature']};">●</span> Target Features</div>
                <div><span style="color: {colors['network_GO']};">●</span> GO Network</div>
                <div><span style="color: {colors['network_Reactome']};">●</span> Reactome</div>
                <div><span style="color: {colors['trajectory']};">●</span> Trajectory</div>
                <div><span style="color: {colors['root']};">●</span> Root Node</div>
            </div>
        </div>
        
        <!-- Instructions -->
        <div style="margin-top: 15px; padding: 10px; background: #e3f2fd; border-radius: 4px; font-size: 11px;">
            <strong>💡 Instructions:</strong>
            <ul style="margin: 5px 0; padding-left: 20px;">
                <li>Click nodes to select (Ctrl+Click for multi-select)</li>
                <li>Double-click to expand neighbors</li>
                <li>Use slider to filter by rank</li>
                <li>Remove floating/isolated nodes</li>
                <li>Remove selected nodes manually</li>
            </ul>
        </div>
    </div>
    """
    
    # Create custom JavaScript
    custom_js = f"""
    <script type="text/javascript">
        // Store full graph data
        const fullGraphData = {full_graph_data};
        
        // Store protected node IDs (features and root)
        const protectedNodes = new Set({json.dumps(feature_ids + ['root'])});
        
        // Track visible nodes and edges
        let visibleNodes = new Set();
        let selectedNodes = new Set();
        const colors = {json.dumps(colors)};
        
        // Initialize visible nodes
        network.body.data.nodes.forEach(node => {{
            visibleNodes.add(node.id);
        }});
        
        // Update rank slider display
        document.getElementById('rank-slider').addEventListener('input', function() {{
            document.getElementById('rank-value').textContent = this.value;
        }});
        
        // Node selection handler
        network.on("selectNode", function(params) {{
            selectedNodes = new Set(params.nodes);
            updateSelectionInfo();
        }});
        
        network.on("deselectNode", function(params) {{
            selectedNodes = new Set(network.getSelectedNodes());
            updateSelectionInfo();
        }});
        
        // Double click to expand
        network.on("doubleClick", function(params) {{
            if (params.nodes.length > 0) {{
                expandNode(params.nodes[0]);
            }}
        }});
        
        function updateSelectionInfo() {{
            const info = document.getElementById('selection-info');
            if (selectedNodes.size === 0) {{
                info.textContent = 'None';
                info.style.color = '#666';
            }} else {{
                const nodeNames = Array.from(selectedNodes).map(id => {{
                    const nodeData = fullGraphData.nodes[id];
                    return nodeData ? nodeData.feature_name : id;
                }}).slice(0, 3).join(', ');
                const extra = selectedNodes.size > 3 ? ` +${{selectedNodes.size - 3}} more` : '';
                info.textContent = nodeNames + extra;
                info.style.color = '#2196F3';
            }}
        }}
        
        function expandNode(nodeId) {{
            console.log('Expanding node:', nodeId);
            
            // Find neighbors in full graph
            const neighbors = fullGraphData.edges
                .filter(e => e.from === nodeId || e.to === nodeId)
                .map(e => e.from === nodeId ? e.to : e.from)
                .filter(n => !visibleNodes.has(n));
            
            if (neighbors.length === 0) {{
                alert('No hidden neighbors found for this node!');
                return;
            }}
            
            // Add neighbor nodes
            const nodesToAdd = [];
            neighbors.forEach(neighborId => {{
                const nodeData = fullGraphData.nodes[neighborId];
                if (!nodeData) return;
                
                // Determine color based on network type
                let color = colors.neighbor;
                if (nodeData.network === 'GO') color = colors.network_GO;
                else if (nodeData.network === 'Reactome') color = colors.network_Reactome;
                
                const label = nodeData.feature_name.length > 15 
                    ? nodeData.feature_name.substring(0, 12) + '...'
                    : nodeData.feature_name;
                
                nodesToAdd.push({{
                    id: neighborId,
                    label: label,
                    title: `<b>${{nodeData.feature_name}}</b><br>` +
                           `network: ${{nodeData.network}}<br>` +
                           `rank: ${{nodeData.rank_index}}<br>` +
                           `<i>Click to expand</i>`,
                    color: color,
                    size: 15,
                    font: {{size: 12, face: 'Times New Roman, Times, serif'}}
                }});
                
                visibleNodes.add(neighborId);
            }});
            
            // Add edges for new nodes
            const edgesToAdd = [];
            fullGraphData.edges.forEach(edge => {{
                if ((visibleNodes.has(edge.from) && neighbors.includes(edge.to)) ||
                    (visibleNodes.has(edge.to) && neighbors.includes(edge.from))) {{
                    edgesToAdd.push({{
                        from: edge.from,
                        to: edge.to,
                        color: {{'color': '#888888'}},
                        arrows: {{'to': {{'enabled': true}}}}
                    }});
                }}
            }});
            
            // Update network
            network.body.data.nodes.add(nodesToAdd);
            network.body.data.edges.add(edgesToAdd);
            
            alert(`Added ${{neighbors.length}} neighbors!`);
        }}
        
        function expandSelectedNodes() {{
            if (selectedNodes.size === 0) {{
                alert('Please select nodes first (click on nodes)');
                return;
            }}
            
            selectedNodes.forEach(nodeId => {{
                expandNode(nodeId);
            }});
        }}
        
        function removeSelectedNodes() {{
            if (selectedNodes.size === 0) {{
                alert('Please select nodes to remove first');
                return;
            }}
            
            const nodesToRemove = Array.from(selectedNodes);
            
            // Also remove all edges connected to these nodes
            const currentEdges = network.body.data.edges.get();
            const edgesToRemove = currentEdges
                .filter(edge => nodesToRemove.includes(edge.from) || nodesToRemove.includes(edge.to))
                .map(edge => edge.id);
            
            console.log('Removing nodes:', nodesToRemove);
            console.log('Removing connected edges:', edgesToRemove.length);
            
            // Remove edges first, then nodes
            if (edgesToRemove.length > 0) {{
                network.body.data.edges.remove(edgesToRemove);
            }}
            network.body.data.nodes.remove(nodesToRemove);
            nodesToRemove.forEach(id => visibleNodes.delete(id));
            selectedNodes.clear();
            updateSelectionInfo();
            
            alert(`Removed ${{nodesToRemove.length}} nodes and ${{edgesToRemove.length}} edges`);
        }}
        
        function removeFloatingNodes() {{
            console.log('=== Removing floating nodes ===');
            
            // Get CURRENT visible nodes and edges from the network (fresh data)
            const currentNodes = network.body.data.nodes.get();
            const currentEdges = network.body.data.edges.get();
            
            console.log('Total current nodes:', currentNodes.length);
            console.log('Total current edges:', currentEdges.length);
            
            // Build set of nodes that have meaningful connections
            // (excluding self-loops and including only edges to OTHER nodes)
            const nodesWithRealEdges = new Set();
            currentEdges.forEach(edge => {{
                // Only count edge if it connects to a DIFFERENT node (not self-loop)
                if (edge.from !== edge.to) {{
                    nodesWithRealEdges.add(edge.from);
                    nodesWithRealEdges.add(edge.to);
                }}
            }});
            
            console.log('Nodes with real edges (excluding self-loops):', nodesWithRealEdges.size);
            console.log('Potential floating:', currentNodes.length - nodesWithRealEdges.size);
            
            // Find floating nodes (nodes with NO edges OR only self-loop edges)
            const floatingNodes = [];
            currentNodes.forEach(node => {{
                // Check if this node has any real edges (not self-loops) in current network
                const hasRealEdges = nodesWithRealEdges.has(node.id);
                
                if (!hasRealEdges) {{
                    // This node is isolated or only has self-loops - check if it's protected
                    const isProtected = protectedNodes.has(node.id) || 
                                       node.type === 'feature' || 
                                       node.type === 'root';
                    
                    if (!isProtected) {{
                        // Check what type of floating node this is for better logging
                        const nodeEdges = currentEdges.filter(e => 
                            e.from === node.id || e.to === node.id
                        );
                        const hasSelfLoop = nodeEdges.some(e => 
                            e.from === node.id && e.to === node.id
                        );
                        
                        const reason = nodeEdges.length === 0 ? 'no edges' : 'only self-loop';
                        floatingNodes.push(node.id);
                        console.log(`✓ Floating node found: ${{node.id}} (${{node.label}}) - ${{reason}} - type: ${{node.type}}`);
                    }} else {{
                        console.log('✗ Skipping protected floating node:', node.id, '(', node.label, ')');
                    }}
                }} 
            }});
            
            console.log('Total floating nodes to remove:', floatingNodes.length);
            
            if (floatingNodes.length === 0) {{
                alert('No floating nodes found!\\nAll nodes are connected to the network.');
                return;
            }}
            
            // Confirm and remove
            if (confirm(`Found ${{floatingNodes.length}} floating (isolated) node(s).\\nThis includes nodes with no edges or only self-loops.\\n\\nRemove them?`)) {{
                network.body.data.nodes.remove(floatingNodes);
                floatingNodes.forEach(id => visibleNodes.delete(id));
                console.log('Successfully removed floating nodes');
                alert(`✅ Removed ${{floatingNodes.length}} floating nodes!`);
            }} else {{
                console.log('User cancelled floating node removal');
            }}
        }}
        
        function applyRankFilter() {{
            const rankCutoff = parseInt(document.getElementById('rank-slider').value);
            console.log('Applying rank filter:', rankCutoff);
            
            // Get CURRENT visible nodes and edges
            const currentNodes = network.body.data.nodes.get();
            const currentEdges = network.body.data.edges.get();
            const currentNodeIds = new Set(currentNodes.map(n => n.id));
            
            // Build a map of existing edges for quick lookup
            const existingEdges = new Set();
            currentEdges.forEach(edge => {{
                existingEdges.add(`${{edge.from}}-${{edge.to}}`);
            }});
            
            const nodesToRemove = [];
            const nodesToAdd = [];
            const edgesToAdd = [];
            
            // Check ALL nodes in visibleNodes (including those we're tracking)
            visibleNodes.forEach(nodeId => {{
                const nodeData = fullGraphData.nodes[nodeId];
                if (!nodeData) return;
                
                const nodeRank = nodeData.rank_index;
                const isCurrentlyVisible = currentNodeIds.has(nodeId);
                const isProtected = protectedNodes.has(nodeId);
                
                // Decide if node should be visible based on rank
                const shouldBeVisible = isProtected || nodeRank <= rankCutoff;
                
                if (shouldBeVisible && !isCurrentlyVisible) {{
                    // Node should be visible but isn't - ADD IT
                    let color = colors.neighbor;
                    if (nodeData.network === 'GO') color = colors.network_GO;
                    else if (nodeData.network === 'Reactome') color = colors.network_Reactome;
                    if (isProtected) color = colors.feature;
                    
                    const label = nodeData.feature_name.length > 15 
                        ? nodeData.feature_name.substring(0, 12) + '...'
                        : nodeData.feature_name;
                    
                    nodesToAdd.push({{
                        id: nodeId,
                        label: label,
                        title: `<b>${{nodeData.feature_name}}</b><br>` +
                               `network: ${{nodeData.network}}<br>` +
                               `rank: ${{nodeData.rank_index}}<br>` +
                               `<i>Click to expand</i>`,
                        color: color,
                        size: isProtected ? 25 : 15,
                        font: {{size: isProtected ? 14 : 12, face: 'Times New Roman, Times, serif'}},
                        type: isProtected ? 'feature' : (nodeData.network === 'GO' ? 'GO' : 'Reactome')
                    }});
                }} else if (!shouldBeVisible && isCurrentlyVisible && !isProtected) {{
                    // Node shouldn't be visible but is - REMOVE IT
                    nodesToRemove.push(nodeId);
                }}
            }});
            
            // After adding nodes, we need to add edges between visible nodes
            if (nodesToAdd.length > 0) {{
                // Add the nodes first
                network.body.data.nodes.add(nodesToAdd);
                
                // Now check for edges involving newly added nodes
                const newNodeIds = new Set(nodesToAdd.map(n => n.id));
                const allVisibleAfterAdd = new Set([...currentNodeIds, ...newNodeIds]);
                
                fullGraphData.edges.forEach(edge => {{
                    const edgeKey = `${{edge.from}}-${{edge.to}}`;
                    // Add edge if both nodes are visible and edge doesn't exist
                    if (allVisibleAfterAdd.has(edge.from) && 
                        allVisibleAfterAdd.has(edge.to) && 
                        !existingEdges.has(edgeKey)) {{
                        edgesToAdd.push({{
                            from: edge.from,
                            to: edge.to,
                            color: {{'color': '#888888'}},
                            arrows: {{'to': {{'enabled': true}}}}
                        }});
                    }}
                }});
                
                // Add edges
                if (edgesToAdd.length > 0) {{
                    network.body.data.edges.add(edgesToAdd);
                }}
            }}
            
            // Remove nodes that don't meet criteria
            if (nodesToRemove.length > 0) {{
                network.body.data.nodes.remove(nodesToRemove);
                nodesToRemove.forEach(id => visibleNodes.delete(id));
            }}
            
            // Show result
            if (nodesToAdd.length > 0 || nodesToRemove.length > 0) {{
                let msg = `Rank filter applied (cutoff: ${{rankCutoff}}):\\n`;
                if (nodesToAdd.length > 0) msg += `Added ${{nodesToAdd.length}} nodes\\n`;
                if (nodesToRemove.length > 0) msg += `Removed ${{nodesToRemove.length}} nodes`;
                alert(msg);
            }} else {{
                alert('No changes - all nodes already match the rank criteria');
            }}
        }}
        
        function resetSelection() {{
            network.unselectAll();
            selectedNodes.clear();
            updateSelectionInfo();
        }}
        
        function resetNetwork() {{
            if (confirm('Reset network to initial state?')) {{
                location.reload();
            }}
        }}
        
        // Initialize
        updateSelectionInfo();
        console.log('interactive network initialized!');
        console.log('Total nodes in full graph:', Object.keys(fullGraphData.nodes).length);
    </script>
    """
    
    # Insert controls and custom JavaScript before </body>
    html_content = html_content.replace('</body>', f'{controls_html}\n{custom_js}\n</body>')
    
    # Write HTML
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
