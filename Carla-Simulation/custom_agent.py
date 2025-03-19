import carla

from agents.navigation.basic_agent import BasicAgent

class CustomAgent(BasicAgent):
    def __init__(self, vehicle, target_speed=100):
        
        """
        :param vehicle: actor to apply to local planner logic onto
        :param target_speed: speed (in Km/h) at which the vehicle will move
        """
        super().__init__(vehicle=vehicle, target_speed=target_speed)

        self.follow_speed_limits(value=False)
        self.ignore_traffic_lights(active=True)
        self.ignore_stop_signs(active=True)
        self.ignore_vehicles(active=True)

    def run_step(self, debug=False):

        """
        Execute one step of navigation.
        :return: carla.VehicleControl
        """
        # velocity = self._vehicle.get_velocity()  
        # speed = 3.6 * (velocity.x**2 + velocity.y**2 + velocity.z**2) ** 0.5  # Convert to km/h
        # print(f"Current speed: {speed:.2f} km/h")
        
        control = self._local_planner.run_step(debug=True)
        return control

# import carla
# from agents.navigation.behavior_agent import BehaviorAgent

# class CustomAgent(BehaviorAgent):
#     def __init__(self, vehicle, behavior='normal'):
#         """
#         :param vehicle: actor to apply to local planner logic onto
#         """
#         super().__init__(vehicle, behavior=behavior)

#         # Disable speed limits and other restrictions
#         self.follow_speed_limits(value=False)
#         self.ignore_traffic_lights(active=True)
#         self.ignore_stop_signs(active=True)
#         self.ignore_vehicles(active=True)

#     def run_step(self, debug=False):
#         """
#         Execute one step of navigation.
#         :return: carla.VehicleControl
#         """
#         velocity = self._vehicle.get_velocity()  
#         speed = 3.6 * (velocity.x**2 + velocity.y**2 + velocity.z**2) ** 0.5  # Convert to km/h
#         print(f"Current speed: {speed:.2f} km/h")
#         control = self._local_planner.run_step(debug=True)
#         return control