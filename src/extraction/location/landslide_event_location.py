import numpy as np
from extraction.location import location

class LandslideEventLocation():
    def __init__(self, candidate_locations):
        self.name = None
        self.lat = None
        self.lng = None
        self.radius = np.inf

        geocoded_locations = []
        points = []
        radiuses = []
        for location_name in candidate_locations:
            lat, lng, radius = location.get_lat_lng_radius(location_name)

            if lat and lng and radius:
                geocoded_locations.append(location_name)
                points.append((lat, lng))
                radiuses.append(radius)

        # centroid = get_centroid(points)
        if radiuses:
            best_location_idx = np.argmin(radiuses)
    
            self.name = geocoded_locations[best_location_idx]
            self.lat = points[best_location_idx][0]
            self.lng = points[best_location_idx][1]
            self.radius = radiuses[best_location_idx]
