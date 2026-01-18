# 3-D seismic velocity model of Dingri area
```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Read data
def load_data(filename):
    """Read data file"""
    data = pd.read_csv(filename, delim_whitespace=True, header=0)
    return data

# Plot slice at specified depth
def plot_depth_slice(data, target_depth, velocity_type='Vp', 
                     resolution_threshold=None, title=None):
    """
    Plot velocity slice at specified depth
    
    Parameters:
    - data: DataFrame containing all data
    - target_depth: Target depth (km), e.g., 0, 5, 10
    - velocity_type: 'Vp' or 'Vs'
    - resolution_threshold: Resolution threshold, data points below this value are not displayed
    - title: Plot title
    """
    
    # Extract data at specified depth
    depth_data = data[data['Z(km)'] == target_depth].copy()
    
    if depth_data.empty:
        print(f"Error: No data found at depth {target_depth} km")
        return None
    
    # Get unique longitude and latitude coordinates
    unique_lons = np.sort(depth_data['Lon.'].unique())
    unique_lats = np.sort(depth_data['Lat.'].unique())
    
    # Create longitude-latitude grid
    lon_grid, lat_grid = np.meshgrid(unique_lons, unique_lats)
    
    # Initialize velocity grid
    velocity_grid = np.full(lon_grid.shape, np.nan)
    
    # Determine velocity column name
    if velocity_type == 'Vp':
        velocity_col = 'Vp(km/s)'
        res_col = 'Vp_resolution'
    elif velocity_type == 'Vs':
        velocity_col = 'Vs(km/s)'
        res_col = 'Vs_resolution'
    else:
        raise ValueError("velocity_type must be 'Vp' or 'Vs'")
    
    # Fill data into grid
    for idx, row in depth_data.iterrows():
        lon_idx = np.where(unique_lons == row['Lon.'])[0][0]
        lat_idx = np.where(unique_lats == row['Lat.'])[0][0]
        
        # Filter data based on resolution threshold
        if resolution_threshold is not None:
            if res_col in row.index and row[res_col] < resolution_threshold:
                continue
        
        velocity_grid[lat_idx, lon_idx] = row[velocity_col]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot contour fill
    contour = ax.contourf(lon_grid, lat_grid, velocity_grid, 
                          levels=50, cmap='jet', alpha=0.8)
    
    # Add contour lines
    contour_lines = ax.contour(lon_grid, lat_grid, velocity_grid, 
                               levels=10, colors='black', linewidths=0.5, alpha=0.5)
    ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%1.2f')
    
    # Add scatter plot showing data point locations
    ax.scatter(depth_data['Lon.'], depth_data['Lat.'], 
               c='white', s=10, alpha=0.3, edgecolors='black', linewidth=0.5)
    
    # Add colorbar
    cbar = plt.colorbar(contour, ax=ax, shrink=0.8)
    cbar.set_label(f'{velocity_type} (km/s)', fontsize=12)
    
    # Set axis labels
    ax.set_xlabel('Longitude (°)', fontsize=12)
    ax.set_ylabel('Latitude (°)', fontsize=12)
    
    # Set title
    if title is None:
        title = f'{velocity_type} Velocity at Depth = {target_depth} km'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Set grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Adjust axis limits
    ax.set_xlim([unique_lons.min(), unique_lons.max()])
    ax.set_ylim([unique_lats.min(), unique_lats.max()])
    
    plt.tight_layout()
    return fig, ax

# Plot multiple depths in subplots
def plot_multiple_depths(data, depths, velocity_type='Vp', 
                         resolution_threshold=None, title_prefix=None):
    """
    Plot slice maps at multiple depths (in subplot form)
    """
    n_depths = len(depths)
    n_cols = min(3, n_depths)
    n_rows = (n_depths + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    
    # If only one subplot, ensure axes is array
    if n_depths == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Determine velocity column name and resolution column name
    if velocity_type == 'Vp':
        velocity_col = 'Vp(km/s)'
        res_col = 'Vp_resolution'
    else:
        velocity_col = 'Vs(km/s)'
        res_col = 'Vs_resolution'
    
    for i, depth in enumerate(depths):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # Extract data at specified depth
        depth_data = data[data['Z(km)'] == depth].copy()
        
        if depth_data.empty:
            ax.text(0.5, 0.5, f'No data at {depth} km', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Depth = {depth} km', fontsize=10)
            continue
        
        # Get unique longitude and latitude coordinates
        unique_lons = np.sort(depth_data['Lon.'].unique())
        unique_lats = np.sort(depth_data['Lat.'].unique())
        
        # Create longitude-latitude grid
        lon_grid, lat_grid = np.meshgrid(unique_lons, unique_lats)
        
        # Initialize velocity grid
        velocity_grid = np.full(lon_grid.shape, np.nan)
        
        # Fill data into grid
        for idx, row in depth_data.iterrows():
            lon_idx = np.where(unique_lons == row['Lon.'])[0][0]
            lat_idx = np.where(unique_lats == row['Lat.'])[0][0]
            
            if resolution_threshold is not None:
                if res_col in row.index and row[res_col] < resolution_threshold:
                    continue
            
            velocity_grid[lat_idx, lon_idx] = row[velocity_col]
        
        # Plot
        contour = ax.contourf(lon_grid, lat_grid, velocity_grid, 
                              levels=50, cmap='jet_r', alpha=0.8)
        
        # Add contour lines
        ax.contour(lon_grid, lat_grid, velocity_grid, 
                   levels=5, colors='black', linewidths=0.5, alpha=0.5)
        
        # Add data points
        ax.scatter(depth_data['Lon.'], depth_data['Lat.'], 
                  c='white', s=5, alpha=0.3, edgecolors='black', linewidth=0.3)
        
        # Set title and labels
        title = f'Depth = {depth} km'
        if title_prefix:
            title = f'{title_prefix}\n{title}'
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Longitude (°)', fontsize=8)
        ax.set_ylabel('Latitude (°)', fontsize=8)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.3)
        ax.set_xlim([unique_lons.min(), unique_lons.max()])
        ax.set_ylim([unique_lats.min(), unique_lats.max()])
    
    # Hide extra subplots
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    
    # Add unified colorbar
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(contour, cax=cbar_ax, label=f'{velocity_type} (km/s)')
    
    if title_prefix:
        fig.suptitle(title_prefix, fontsize=14, fontweight='bold', y=0.98)
    
    return fig, axes

# Display data statistics
def show_data_info(data):
    """Display basic statistical information of data"""
    print("Basic Data Information:")
    print("=" * 50)
    print(f"Number of data rows: {len(data)}")
    print(f"Number of data columns: {len(data.columns)}")
    print(f"Column names: {list(data.columns)}")
    print(f"Depth range: {data['Z(km)'].min()} - {data['Z(km)'].max()} km")
    print(f"Longitude-Latitude range:")
    print(f"  Longitude: {data['Lon.'].min()} - {data['Lon.'].max()}")
    print(f"  Latitude: {data['Lat.'].min()} - {data['Lat.'].max()}")
    print(f"Vp velocity range: {data['Vp(km/s)'].min()} - {data['Vp(km/s)'].max()} km/s")
    print(f"Vs velocity range: {data['Vs(km/s)'].min()} - {data['Vs(km/s)'].max()} km/s")
    print(f"Vp resolution range: {data['Vp_resolution'].min()} - {data['Vp_resolution'].max()}")
    print(f"Vs resolution range: {data['Vs_resolution'].min()} - {data['Vs_resolution'].max()}")
    print("\nData distribution at different depths:")
    depth_counts = data['Z(km)'].value_counts().sort_index()
    for depth, count in depth_counts.items():
        print(f"  Depth {depth} km: {count} data points")
    
    return depth_counts

# Usage example
if __name__ == "__main__":
    # Usage instructions:
    # 1. Save your data as a text file, e.g., 'velocity_data.txt'
    # 2. Ensure column name format: Lon.    Lat.    Z(km)  Vp(km/s)  Vp_resolution   Vs(km/s)  Vs_resolution
    # Read data
    data = load_data('Dingri-model3d-Vp-Vs.txt')  # Replace with your file name
    
    # Display data information
    depth_counts = show_data_info(data)
    
    # Get all depths
    all_depths = depth_counts.index.tolist()
    
    print("\nAvailable depths:", all_depths)
    
    # Example: Plot Vp velocity at multiple depths
    print("\n3. Plot Vp velocity at multiple depths:")
    fig3, axes3 = plot_multiple_depths(data, depths=[0, 5, 10, 15, 20, 30], 
                                       velocity_type='Vp', resolution_threshold=0.85,
                                       title_prefix='Vp Velocity Distribution')
    plt.show()
    
    fig4, axes4 = plot_multiple_depths(data, depths=[0, 5, 10, 15, 20, 30], 
                                       velocity_type='Vs', resolution_threshold=0.85,
                                       title_prefix='Vs Velocity Distribution')
    plt.show()
```
