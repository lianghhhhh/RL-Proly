class ObservationProcessor:
    def __init__(self, observation_structure):
        self.observation_structure = observation_structure
        self.observation_size = self._calculate_observation_size(observation_structure)
        print(f"Observation size calculated: {self.observation_size}")

    def get_size(self):
        return self.observation_size

    def _calculate_observation_size(self, observation_structure):
        total_size = 0

        for item in observation_structure:
            item_type = item.get("type", "")
            item_key = item.get("key", "")

            if item_key == "flattened":
                vector_size = item.get("vector_size", 0)
                return vector_size

            if item_type == "Vector3":
                total_size += 3
            elif item_type == "Vector2":
                total_size += 2
            elif item_type == "float" or item_type == "int" or item_type == "bool":
                total_size += 1
            elif item_type == "Grid":
                grid_size = item.get("grid_size", 0)
                sub_items = item.get("items", [])
                sub_item_size = self._calculate_observation_size(sub_items)
                total_size += sub_item_size * grid_size * grid_size
            elif item_type == "List":
                sub_items = item.get("items", [])
                sub_item_size = self._calculate_observation_size(sub_items)
                sub_item_count = item.get("item_count", 0)

                if sub_item_count > 0:
                    total_size += sub_item_size * sub_item_count
                else:
                    total_size += sub_item_size

        return total_size