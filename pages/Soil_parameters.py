import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

import folium
from streamlit_folium import st_folium

from typing import Dict, List, Tuple

try:
    import rasterio
    from rasterio.transform import rowcol
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False
    print("Warning: rasterio not installed. Install with: pip install rasterio")


def create_multi_axis_depth_profile(df, properties_to_plot=None):
    """
    Create a depth profile chart where each property has its own X-axis scale.
    All properties are plotted on the same chart but with different X-axis ranges.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with columns: 'Depth', 'Clay', 'Sand', 'C org', etc.
    properties_to_plot : list, optional
        List of column names to plot. If None, plots all numeric columns except 'Depth'
    
    Returns:
    --------
    plotly.graph_objects.Figure
    """
    
    # Depth to numeric mapping (middle of depth range in cm)
    depth_mapping = {
        '0-5cm': 2.5,
        '5-15cm': 10,
        '15-30cm': 22.5,
        '30-60cm': 45,
        '60-100cm': 80,
        '100-200cm': 150
    }
    
    # Create numeric depth column
    df_plot = df.copy()
    df_plot['depth_numeric'] = df_plot['Depth'].map(depth_mapping)
    
    # Determine which properties to plot
    if properties_to_plot is None:
        properties_to_plot = [col for col in df.columns 
                             if col not in ['Depth', 'depth_numeric']]
    
    # Color palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Create figure
    fig = go.Figure()
    
    # Calculate normalized positions (0-1 range) for each property
    # This allows each to use its own scale while appearing on same plot
    for i, prop in enumerate(properties_to_plot):
        color = colors[i % len(colors)]
        
        # Get min and max for this property
        values = df_plot[prop].dropna()
        if len(values) == 0:
            continue
            
        min_val = values.min()
        max_val = values.max()
        
        # Normalize to 0-1 range, then scale to property's position
        # Each property gets equal horizontal space
        n_props = len(properties_to_plot)
        prop_width = 1.0 / n_props
        prop_start = i * prop_width
        prop_end = (i + 1) * prop_width
        
        # Normalize values to 0-1, then scale to property's space
        if max_val != min_val:
            normalized = (values - min_val) / (max_val - min_val)
        else:
            normalized = pd.Series([0.5] * len(values), index=values.index)
        
        x_scaled = prop_start + normalized * prop_width * 0.9  # 0.9 to leave gaps
        
        # Add trace
        fig.add_trace(go.Scatter(
            x=x_scaled,
            y=df_plot['depth_numeric'],
            mode='lines+markers',
            name=prop,
            line=dict(color=color, width=2),
            marker=dict(size=10, color=color),
            hovertemplate=(
                f'<b>{prop}</b><br>' +
                'Value: %{customdata:.2f}<br>' +
                'Depth: %{text}<br>' +
                '<extra></extra>'
            ),
            text=df_plot['Depth'],
            customdata=values,  # Store original values for hover
            showlegend=False
        ))
        
        # Add invisible scatter points at min and max for axis ticks
        # This helps create proper axis reference
        fig.add_trace(go.Scatter(
            x=[prop_start + 0.05, prop_start + prop_width * 0.9],
            y=[df_plot['depth_numeric'].max(), df_plot['depth_numeric'].max()],
            mode='markers',
            marker=dict(size=0.1, color=color, opacity=0),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Create custom X-axis annotations for each property
    annotations = []
    shapes = []
    
    for i, prop in enumerate(properties_to_plot):
        color = colors[i % len(colors)]
        values = df_plot[prop].dropna()
        if len(values) == 0:
            continue
            
        min_val = values.min()
        max_val = values.max()
        
        n_props = len(properties_to_plot)
        prop_width = 1.0 / n_props
        prop_center = i * prop_width + prop_width / 2
        prop_start = i * prop_width
        prop_end = i * prop_width + prop_width * 0.9
        
        # Add property label
        annotations.append(dict(
            x=prop_center,
            y=-0.15,
            xref='x',
            yref='paper',
            text=f'<b>{prop}</b>',
            showarrow=False,
            font=dict(size=12, color=color),
            xanchor='center'
        ))
        
        # Add min value
        annotations.append(dict(
            x=prop_start + 0.05,
            y=-0.07,
            xref='x',
            yref='paper',
            text=f'{min_val:.1f}',
            showarrow=False,
            font=dict(size=10, color=color),
            xanchor='center'
        ))
        
        # Add max value
        annotations.append(dict(
            x=prop_end,
            y=-0.07,
            xref='x',
            yref='paper',
            text=f'{max_val:.1f}',
            showarrow=False,
            font=dict(size=10, color=color),
            xanchor='center'
        ))
        
        # Add axis line for this property
        shapes.append(dict(
            type='line',
            xref='x',
            yref='paper',
            x0=prop_start + 0.08,
            x1=prop_end-0.03,
            y0=-0.02,
            y1=-0.02,
            line=dict(color=color, width=1)
        ))
        
        # Add tick marks
        # shapes.append(dict(
        #     type='line',
        #     xref='x',
        #     yref='paper',
        #     x0=prop_start + 0.05,
        #     x1=prop_start + 0.05,
        #     y0=-0.05,
        #     y1=-0.03,
        #     line=dict(color=color, width=2)
        # ))
        # shapes.append(dict(
        #     type='line',
        #     xref='x',
        #     yref='paper',
        #     x0=prop_end,
        #     x1=prop_end,
        #     y0=-0.05,
        #     y1=-0.03,
        #     line=dict(color=color, width=2)
        # ))
    
    # Update layout
    fig.update_layout(
        xaxis=dict(
            range=[0, 1],
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            showline=False
        ),
        yaxis=dict(
            autorange="reversed",
            title="Depth (cm)",
            showgrid=True,
            gridcolor='#e5e5e5',
            zeroline=True
        ),
        hovermode='closest',
        # legend=dict(),
        # legend=dict(
        #     orientation="h",
        #     yanchor="top",
        #     y=1.1,
        #     xanchor="right",
        #     x=1,
        #     bgcolor="rgba(255, 255, 255, 0.8)",
        #     bordercolor="lightgray",
        #     borderwidth=1
        # ),
        height=200,
        template="plotly_white",
        margin=dict(l=0, r=0, t=0, b=40),  # Extra bottom margin for axes
        annotations=annotations,
        shapes=shapes
    )
    
    return fig

st.header(":material/landslide: Soil parameters")

st.markdown(
    "Processing of cosmic-ray neutron data requires information about additional hydrogen pools in the soil. Here you can estimate the additional water equivalent based on soil parameters."
)


    
class SoilGridsFast:
    """Fast SoilGrids queries using direct raster file access."""
    
    BASE_URL = "https://files.isric.org/soilgrids/latest/data"
    
    # Property definitions with conversion factors
    PROPERTIES = {
        'bdod': {
            'name': 'Bulk density',
            'unit': 'g/cm³',
            'conversion': 0.01,  # cg/cm³ to g/cm³
            'path': 'bdod'
        },
        'clay': {
            'name': 'Clay',
            'unit': 'g/kg',
            'conversion': 1.0,  # Already in g/kg
            'path': 'clay'
        },
        'sand': {
            'name': 'Sand',
            'unit': 'g/kg',
            'conversion': 1.0,  # Already in g/kg
            'path': 'sand'
        },
        'soc': {
            'name': 'C org.',
            'unit': 'g/kg',
            'conversion': 0.1,  # dg/kg to g/kg
            'path': 'soc'
        },
    }
    
    DEPTHS = ['0-5cm', '5-15cm', '15-30cm']
    
    def __init__(self):
        """Initialize the fast SoilGrids client."""
        if not RASTERIO_AVAILABLE:
            raise ImportError("rasterio is required. Install with: pip install rasterio")
    
    def _get_vrt_url(self, property_code: str, depth: str, stat: str = 'mean') -> str:
        """
        Construct URL for VRT file.
        
        Args:
            property_code: Property code (e.g., 'clay', 'sand')
            depth: Depth layer (e.g., '0-5cm')
            stat: Statistic ('mean', 'Q0.05', 'Q0.5', 'Q0.95', 'uncertainty')
        
        Returns:
            URL to the VRT file
        """
        prop_info = self.PROPERTIES[property_code]
        return f"{self.BASE_URL}/{prop_info['path']}/{property_code}_{depth}_{stat}.vrt"
    
    def query_point(self, lat: float, lon: float,
                    properties: List[str] = None,
                    depths: List[str] = None,
                    stat: str = 'mean') -> Dict:
        """
        Query soil properties for a single point.
        This is MUCH faster than the REST API!
        
        Args:
            lat: Latitude (decimal degrees WGS84, -90 to 90)
            lon: Longitude (decimal degrees WGS84, -180 to 180)
            properties: List of property codes
            depths: List of depth layers
            stat: Statistic to retrieve ('mean' is default)
        
        Returns:
            Dictionary with soil properties by depth
        """
        from rasterio.warp import transform as warp_transform
        
        # Set defaults
        if properties is None:
            properties = ['bdod', 'clay', 'sand', 'soc']
        if depths is None:
            depths = ['0-5cm', '5-15cm', '15-30cm']
        
        # Validate inputs
        if not -90 <= lat <= 90:
            raise ValueError(f"Latitude must be between -90 and 90, got {lat}")
        if not -180 <= lon <= 180:
            raise ValueError(f"Longitude must be between -180 and 180, got {lon}")
        
        results = {
            'location': {'latitude': lat, 'longitude': lon},
            'layers': {}
        }
        
        # Query each property and depth combination
        for depth in depths:
            if depth not in results['layers']:
                results['layers'][depth] = {}
            
            for prop in properties:
                try:
                    vrt_url = self._get_vrt_url(prop, depth, stat)
                    
                    # Open raster and sample at point
                    with rasterio.open(vrt_url) as src:
                        # CRITICAL FIX: Transform coordinates from WGS84 to raster CRS
                        # SoilGrids uses Homolosine projection, not WGS84!
                        xs, ys = warp_transform('EPSG:4326', src.crs, [lon], [lat])
                        x_proj, y_proj = xs[0], ys[0]
                        
                        # Get pixel coordinates using transformed coordinates
                        row, col = rowcol(src.transform, x_proj, y_proj)
                        
                        # Check if point is within bounds
                        if 0 <= row < src.height and 0 <= col < src.width:
                            # Read single pixel value
                            window = ((row, row + 1), (col, col + 1))
                            value = src.read(1, window=window)[0, 0]
                            
                            # Convert nodata to None
                            if value == src.nodata or np.isnan(value):
                                value = None
                            else:
                                # Apply conversion factor
                                value = float(value) * self.PROPERTIES[prop]['conversion']
                        else:
                            value = None
                    
                    results['layers'][depth][prop] = {
                        'value': value,
                        'unit': self.PROPERTIES[prop]['unit'],
                        'description': self.PROPERTIES[prop]['name']
                    }
                
                except Exception as e:
                    print(f"Warning: Could not retrieve {prop} at {depth}: {e}")
                    results['layers'][depth][prop] = {
                        'value': None,
                        'unit': self.PROPERTIES[prop]['unit'],
                        'description': self.PROPERTIES[prop]['name']
                    }
        
        return results
    
    def to_dataframe(self, results: Dict) -> pd.DataFrame:
        """
        Convert query results to a pandas DataFrame.
        
        Args:
            results: Results dictionary from query_point()
        
        Returns:
            DataFrame with soil properties by depth
        """
        rows = []
        
        for depth, properties in results['layers'].items():
            row = {'Depth': depth}
            
            for prop_code, prop_data in properties.items():
                value = prop_data['value']
                desc = prop_data['description']
                unit = prop_data['unit']
                col_name = f"{desc} ({unit})"
                row[col_name] = value
            
            rows.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(rows)
        
        # Reorder columns with Depth first
        cols = ['Depth'] + [col for col in df.columns if col != 'Depth']
        df = df[cols]
        
        return df
    
    def query_multiple_points(self, coordinates: List[Tuple[float, float]],
                             properties: List[str] = None,
                             depths: List[str] = None) -> pd.DataFrame:
        """
        Query multiple points and return combined DataFrame.
        
        Args:
            coordinates: List of (lat, lon) tuples
            properties: List of property codes
            depths: List of depth layers
        
        Returns:
            DataFrame with all points and depths
        """
        all_results = []
        
        for i, (lat, lon) in enumerate(coordinates):
            print(f"Querying point {i+1}/{len(coordinates)}: ({lat:.4f}, {lon:.4f})")
            
            try:
                results = self.query_point(lat, lon, properties, depths)
                df = self.to_dataframe(results)
                df.insert(0, 'Point_ID', i + 1)
                df.insert(1, 'Latitude', lat)
                df.insert(2, 'Longitude', lon)
                all_results.append(df)
            except Exception as e:
                print(f"  Error: {e}")
                continue
        
        if all_results:
            return pd.concat(all_results, ignore_index=True)
        else:
            return pd.DataFrame()
        
def get_soil_data(lat, lon):
    """Main function to demonstrate usage."""
    
    # Example coordinates (you can modify these)
    # Default: Wageningen, Netherlands
    latitude = lat #51.9851
    longitude = lon #5.6642
    
    # Initialize API client
    # api = SoilGridsAPI()
    api = SoilGridsFast()
    
    print(f"\nQuerying SoilGrids for location: {latitude}°N, {longitude}°E")
    print("Please wait...\n")
    
    try:
        with st.spinner("Querrying SoilGrids API, just a moment...", show_time=True):
            
            results = api.query_point(
                lat=latitude,
                lon=longitude,
                properties=['bdod', 'clay', 'sand', 'soc'],
                depths=['0-5cm', '5-15cm', '15-30cm']
            )
            
            # Convert to DataFrame
            df = api.to_dataframe(results)
        
            
        st.toast("Data has been received!", icon=":material/check:")
        return df

    except Exception as e:
        print(f"\nError: {e}")
        # st.error(f"Error querying SoilGrids API: {e}")
        return None


# Initialize session state for coordinates if not exists
if 'latitude' not in st.session_state:
    st.session_state.latitude = 51.9851
if 'longitude' not in st.session_state:
    st.session_state.longitude = 5.6642
if 'map_clicked' not in st.session_state:
    st.session_state.map_clicked = False

@st.fragment
def show_map():
    # Create two columns for layout
    col1, col2 = st.columns(2)

    with col1: 
        # Number inputs for manual coordinate entry
        new_lat = st.number_input(
            "Latitude",
            min_value=-90.0,
            max_value=90.0,
            value=st.session_state.latitude,
            step=0.0001,
            format="%.4f",
            key="lat_input"
        )
        
        new_lon = st.number_input(
            "Longitude",
            min_value=-180.0,
            max_value=180.0,
            value=st.session_state.longitude,
            step=0.0001,
            format="%.4f",
            key="lon_input"
        )
    
        # Check if manual input changed
        if (new_lat != st.session_state.latitude or 
            new_lon != st.session_state.longitude):
            st.session_state.latitude = new_lat
            st.session_state.longitude = new_lon
            st.session_state.map_clicked = False
            st.rerun(scope="fragment")
    
        # Create folium map centered on current coordinates
        soil_property = st.pills(
            "Display soil property",
            options=[
                "Clay",
                "Sand", 
                "Silt",
                "C org.",
                "Bulk density",
            ],
            help="Data is provided by ISRIC - World Soil Information through the [SoilGrids project](https://soilgrids.org) (see also [Poggio et al., 2021](https://doi.org/10.5194/soil-7-217-2021))"
        )

        soil_depth = st.pills(
            "Display depth layer (cm)",
            options=["0-5", "5-15", "15-30"],
            default="5-15"

        )

    with col2:
        m = folium.Map(
            location=[st.session_state.latitude, st.session_state.longitude],
            zoom_start=10,
            tiles='OpenStreetMap'
        )
        property_mapping = {
            "Clay": "clay",
            "Sand": "sand",
            "C org.": "soc",
            "Bulk density": "bdod",
            "Silt": "silt"
        }
        
        if soil_property is not None:
            prop_code = property_mapping[soil_property]
            
            # SoilGrids WMS endpoint
            wms_url = f"https://maps.isric.org/mapserv?map=/map/{prop_code}.map"
            
            # Layer name format: property_depth_statistic
            # Note: depths use underscores in WMS
            # depth_wms = soil_depth.replace("cm", "cm").replace("-", "_")
            layer_name = f"{prop_code}_{soil_depth}cm_mean"
            
            # Add WMS layer
            folium.raster_layers.WmsTileLayer(
                url=wms_url,
                layers=layer_name,
                name=f"{soil_property} ({soil_depth} cm)",
                fmt="image/png",
                # transparent=True,
                # opacity=0.5,
                overlay=True,
                control=True,
            ).add_to(m)

        # Add a marker at the current location
        folium.Marker(
            [st.session_state.latitude, st.session_state.longitude],
            tooltip=f"Lat: {st.session_state.latitude:.4f}<br>Lon: {st.session_state.longitude:.4f}°",
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(m)

        folium.LayerControl().add_to(m)
        
        # Display the map and capture click events
        map_data = st_folium(
            m,
            height=300,
            key="map"
        )
        
        # Check if map was clicked
        if map_data and map_data.get('last_clicked'):
            clicked_lat = map_data['last_clicked']['lat']
            clicked_lon = map_data['last_clicked']['lng']
            
            # Only update if coordinates actually changed
            if (clicked_lat != st.session_state.latitude or 
                clicked_lon != st.session_state.longitude):
                st.session_state.latitude = clicked_lat
                st.session_state.longitude = clicked_lon
                st.session_state.map_clicked = True
                st.rerun(scope="fragment")

with st.container(border=True):
    st.subheader("Location")
    show_map()


# c1, c2, c3 = st.columns(3, vertical_alignment="bottom")
# lat = c1.number_input(value=51.9851, step=0.01, format="%.06f", label="Latitude (°)")
# lon = c2.number_input(value=5.6642, step=0.01, format="%.06f", label="Longitude (°)")

st.session_state["clay"] = 0.0
st.session_state["corg"] = 0.0

with st.container(border=True):
    st.subheader("SoilGrids data", help="Data is provided by ISRIC - World Soil Information through the [SoilGrids project](https://soilgrids.org) (see also [Poggio et al., 2021](https://doi.org/10.5194/soil-7-217-2021))")

    if st.button(":material/cloud_download: Fetch SoilGrids data at this point", type="primary"):
        df = get_soil_data(st.session_state.latitude, st.session_state.longitude)
        st.session_state["df"] = df

    if "df" in st.session_state:
        result_c1, result_c2 = st.columns(2)
        result_c1.dataframe(st.session_state["df"])
        result_c1.write("For further analysis, it is suggested to take the mean of the upper 15 cm.")
        st.session_state["clay"] = st.session_state["df"].iloc[0:2]["Clay (g/kg)"].mean() /1000
        st.session_state["corg"] = st.session_state["df"].iloc[0:2]["C org. (g/kg)"].mean() /1000

        fig = create_multi_axis_depth_profile(st.session_state["df"])
        result_c2.plotly_chart(fig, use_container_width=True)


def Greacen1981(clay_gg):
    lw = clay_gg*0.1783
    return lw

def DongOchsner2018(clay_gg):
    lw = clay_gg*0.16 + 0.0149
    return lw

@st.fragment
def conversion_lw():
    c1, c2, c3 = st.columns(3)
    with c1:
        clay_content = st.number_input(
            "Average Clay content (g/g)",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state["clay"],
            step=0.01,
            format="%.2f",
            key="clay_input"
        )
    with c2:
        lw_method = st.radio("Method", options=["Greacen (1981)", "Dong & Ochsner (2018)"], index=0, help="Given Clay (g/g) as $C$, then the lattice water (g/g) is:\n- $C \cdot 0.1783$ (Greacen, 1981)\n- $C\cdot 0.16 + 0.0149$ (Dong & Ochsner, 2018)\n")
    with c3:
        if lw_method == "Greacen (1981)":
            f = Greacen1981
        elif lw_method == "Dong & Ochsner (2018)":
            f = DongOchsner2018
        lw = f(clay_content)
        st.metric("Lattice water (g/g)", value=f"{lw:.3f}")


with st.container(border=True):
    st.subheader("Conversion to lattice water")
    conversion_lw()

def Franz2013(soc_gg):
    owe = soc_gg * 0.556
    return owe

@st.fragment
def conversion_owe():
    c1, c2, c3 = st.columns(3)
    with c1:
        organic_carbon = st.number_input(
            "Average soil organic carbon (g/g)",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state["corg"],
            step=0.01,
            format="%.2f",
            key="corg_input"
        )
    with c2:
        owe_method = st.radio("Method", options=["Franz et al. (2013)"], index=0, help="Given soil organic carbon (g/g) as $O$, then the organic water equivalent (g/g) is:\n- $O \cdot 0.556$ (Franz et al., 2013)")
    with c3:
        if owe_method == "Franz et al. (2013)":
            f = Franz2013
        owe = f(organic_carbon)
        st.metric("Soil organic water equivalent (g/g)", value=f"{owe:.3f}")


with st.container(border=True):
    st.subheader("Conversion to organic carbon water equivalent")
    conversion_owe()


# show_footer()