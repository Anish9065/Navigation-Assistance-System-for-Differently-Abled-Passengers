class NavigationAssistant:
    def __init__(self):
        """
        Initializes the Navigation Assistant logic.
        """
        # A simple mapping of destinations for demonstration
        self.destinations = {
            "platform_1": {"description": "Platform 1", "accessible_route": "Use Elevator A to reach Platform 1."},
            "platform_2": {"description": "Platform 2", "accessible_route": "Use Elevator B to reach Platform 2."},
            "ticket_counter": {"description": "Ticket Counter", "accessible_route": "Proceed straight to the main hall for Ticket Counter."},
            "restroom": {"description": "Restroom", "accessible_route": "Accessible restroom is located near the eastern exit."},
            "exit": {"description": "Exit", "accessible_route": "Main exit is straight ahead."}
        }

    def get_navigation_help(self, person_type, destination_key):
        """
        Provides modular navigation help based on person type and destination.

        person_type: 'normal_person', 'wheelchair_user', 'blind_person', 'crutch_user'
        destination_key: string key for destination
        """
        if destination_key not in self.destinations:
            return f"Sorry, destination '{destination_key}' is unknown."

        destination_info = self.destinations[destination_key]
        dest_name = destination_info["description"]
        route = destination_info["accessible_route"]

        guidance_message = ""

        if person_type == "wheelchair_user":
            guidance_message = f"Wheelchair priority route to {dest_name}: {route}. Ramps are available along the way."
        elif person_type == "blind_person":
            guidance_message = f"Audio guidance to {dest_name}: Proceed slowly. {route} Use tactile paving on the floor."
        elif person_type == "crutch_user":
            guidance_message = f"Accessible route to {dest_name}: {route}. Please use the elevators."
        else:
            # Normal person
            guidance_message = f"Navigating to {dest_name}. Proceed straight ahead."

        return guidance_message

    def get_spatial_warning(self, object_x_center, frame_width):
        """
        Provides spatial warning based on object position in the frame.
        """
        left_bound = frame_width * 0.33
        right_bound = frame_width * 0.67

        if object_x_center < left_bound:
            return "on your left"
        elif object_x_center > right_bound:
            return "on your right"
        else:
            return "straight ahead"

if __name__ == "__main__":
    nav = NavigationAssistant()
    print(nav.get_navigation_help("wheelchair_user", "platform_1"))
