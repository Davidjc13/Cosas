import argparse
import pandas as pd

MOP_COLUMN = "MOP"
TRACKED_ID_COLUMN = "Tracked ID"
CORE_FRAME_COLUMN = "frame_number"
TRACKED_FRAME_COLUMN = "Frame"
FRAME_COLUMN = "Frame"
MAX_DISTANCE_FRAMES = 60
X_MIN = "X Min"
Y_MIN = "Y Min"
X_MAX = "X Max"
Y_MAX = "Y Max"

DISTANCE_THRESHOLD = 0.005


class TrackFixer:
    """Class to fix duplicates tracked ids from CORE.csv. Evaluate if two signals with same MOP and different tracked id are the same
        
        Distance function is a function that receives two bounding boxes and returns a distance between them. In the form of:
        
        distance_function(first_bounding_box, second_bounding_box:) -> float: 
        Where first_bounding_box and second_bounding_box are tuples containing the bounding box coordinates with the following format:
        (x_min, y_min, x_max, y_max)
        
        Coordinates must be normalized between 0 and 1.
    """
    def __init__(self, tracked_predictions_dataframe, core_dataframe, distance_function = None , output_path=None):
        """
        Construction of TrackFixer class.
        Args:
            - tracked_predictions_dataframe (pd.DataFrame): Tracked predicitions csv with differents tracked ids
            - core_dataframe (pd.DataFrame): Core.csv dataframe
            - distance_function (Callable): Function to calculate distance between two bounding boxes. (default: None)
            - output_path (str): Path to save revised CORE.csv (default: None)
        """
        self.distance_function = distance_function
        self.tracked_predictions_dataframe = tracked_predictions_dataframe[
            tracked_predictions_dataframe[TRACKED_ID_COLUMN] != -1
        ].copy()
        self.core_dataframe = core_dataframe
        self.tracked_ids_list = self.core_dataframe[TRACKED_ID_COLUMN].tolist()

    def possible_tracked_ids(self, mop):
        """Method to return possible match given mop (signal/road damage class)

        Args:
            - mop (str): Signal/Road Damage class.

        Returns:
            list[int]: List containing differents tracked ids.
        """
        core_filtered_by_mop = self.core_dataframe[
            self.core_dataframe[MOP_COLUMN] == mop
        ]
        return core_filtered_by_mop[TRACKED_ID_COLUMN].tolist()

    def _retrieve_frames(self, tracked_id):
        """Given a tracked id returns all frames appareances

        Args:
            tracked_id (int): Tracked id to filter.
        """

        frames_appareances = self.tracked_predictions_dataframe[
            self.tracked_predictions_dataframe[TRACKED_ID_COLUMN] == tracked_id
        ][TRACKED_FRAME_COLUMN]

        frames_appareances_list = frames_appareances.tolist()

        frames_appareances_set = set(frames_appareances_list)

        return frames_appareances_set

    @staticmethod
    def _minor_difference(tracked_id_frames, suspected_match_frames):
        """Retrieve the minor difference using set operations.

        Args:
            - tracked_id_frames (set(int)): Set containing all frames associated to tracked_id.
            - suspected_match_frames (set(int)): Set containing all frames associated to possible match.

        Returns:
            tuple(int,int): Frame associated to first set and associated to second set.
        """

        unique_tracked_id_frames = tracked_id_frames.difference(suspected_match_frames)
        unique_suspected_match_frames = suspected_match_frames.difference(
            tracked_id_frames
        )

        filtered_tracked_id_frames = unique_tracked_id_frames
        filtered_suspected_match_frames = unique_suspected_match_frames

        tracked_id_frame, closest_frame = None, None
        minimal_distance = float("inf")

        for frame in sorted(filtered_tracked_id_frames):

            for suspected_frame in sorted(filtered_suspected_match_frames):

                distance = abs(frame - suspected_frame)

                if distance < minimal_distance:

                    tracked_id_frame, closest_frame = frame, suspected_frame
                    minimal_distance = distance

        return tracked_id_frame, closest_frame

    def _bounding_box_distance(self, tracked_id_row, suspected_row):
        """Given two predictions returns bounding box distance. If distance function is not provided when class is initialized, it will use the default one.
        Default function is the average distance between min and max coordinates.

        Args:
            tracked_id_row (pd.Series): Series containing bounding box columns.
            suspected_row (pd.Series): Series containing bounding box columns.
            distance_fucntion (function): Function to calculate distance between two bounding boxes. (default: None)
            
        Returns:
            float: Distance of bounding box (Average distance between minimun coordinates and max.)
        """

        if not tracked_id_row.empty and not suspected_row.empty:

            first_bounding_box = (
                tracked_id_row[X_MIN].iloc[0],
                tracked_id_row[Y_MIN].iloc[0],
                tracked_id_row[X_MAX].iloc[0],
                tracked_id_row[Y_MAX].iloc[0],
            )

            second_bounding_box = (
                suspected_row[X_MIN].iloc[0],
                suspected_row[Y_MIN].iloc[0],
                suspected_row[X_MAX].iloc[0],
                suspected_row[Y_MAX].iloc[0],
            )

            min_bounding_box_distance = abs(
                first_bounding_box[0] - second_bounding_box[0]
            ) + abs(first_bounding_box[1] - second_bounding_box[1])

            max_bounding_box_distance = abs(
                first_bounding_box[2] - second_bounding_box[2]
            ) + abs(first_bounding_box[3] - second_bounding_box[3])

            if distance_fucntion:
                bounding_box_distance = distance_fucntion(first_bounding_box, second_bounding_box)
                 
            else:
                bounding_box_distance = (
                    min_bounding_box_distance + max_bounding_box_distance
                ) * 0.5


            return bounding_box_distance

        return None

    @staticmethod
    def _filter_dataframe(tracked_predictions_dataframe, tracked_id_row, frame):
        """
        Method to return the most similar row to a given one.

        Args:
            tracked_predictions_dataframe (pd.DataFrame): Dataframe with all predictions.
            tracked_id_row (pd.Series): Row to compare with.
            frame (int): Frame number for suspected dataframe.

        Returns:
            pd.Series: The most similar row.
        """

        tracked_predictions_dataframe = tracked_predictions_dataframe[
            (
                tracked_predictions_dataframe[MOP_COLUMN]
                == tracked_id_row[MOP_COLUMN].iloc[0]
            )
            & (tracked_predictions_dataframe[TRACKED_FRAME_COLUMN] == frame)
        ]

        if len(tracked_predictions_dataframe) > 1:

            if tracked_id_row[X_MIN].iloc[0] > 0.5:
                tracked_predictions_dataframe = tracked_predictions_dataframe[
                    tracked_predictions_dataframe[X_MIN] > 0.5
                ]

            else:
                tracked_predictions_dataframe = tracked_predictions_dataframe[
                    tracked_predictions_dataframe[X_MIN] < 0.5
                ]

        return tracked_predictions_dataframe

    def _is_same_object(self, tracked_id_row, suspected_match_row):
        """
        Method that given two rows returns if object could be the same.

        Args:
            tracked_id_row (pd.Series): Row containing object
            suspected_match_row (pd.Series): Row that is suspected to contain same object

        Returns:
            bool: True if object are the same.
        """

        distance = self._bounding_box_distance(tracked_id_row, suspected_match_row)
        if distance:
            frame_distance = abs(
                tracked_id_row[FRAME_COLUMN].iloc[0]
                - suspected_match_row[FRAME_COLUMN].iloc[0]
            )

            if (
                distance <= DISTANCE_THRESHOLD * frame_distance
                and frame_distance < MAX_DISTANCE_FRAMES
            ):
                return True

        return False

    @staticmethod
    def retrieve_tracked_id_row(tracked_predictions, tracked_id, frame):
        """Retrieves row associated to frame and tracked_id from tracked_predictions

        Args:
            tracked_predictions (pd.DataFrame):
            tracked_id (int): _description_
            frame (int): _description_

        Returns:
            pd.Series: _description_
        """
        tracked_id_row = tracked_predictions[
            (tracked_predictions[TRACKED_ID_COLUMN] == tracked_id)
            & (tracked_predictions[FRAME_COLUMN] == frame)
        ]

        return tracked_id_row

    @staticmethod
    def _update_dataframe(dataframe, old_tracked_id, new_tracked_id):
        """Updates dataframe and assign new tracked id.

        Args:
            dataframe (pd.DataFrame): Dataframe to update
            old_tracked_id (int): Tracked ID to be updated
            new_tracked_id (int): New Tracked ID.

        Returns:
            pd.DataFrame: Updated dataframe
        """

        updated_dataframe = dataframe

        updated_dataframe.loc[
            updated_dataframe[TRACKED_ID_COLUMN] == old_tracked_id, "old tracked id"
        ] = old_tracked_id
        updated_dataframe.loc[
            updated_dataframe[TRACKED_ID_COLUMN] == old_tracked_id, TRACKED_ID_COLUMN
        ] = new_tracked_id

        return updated_dataframe

    def main(self):
        """Method to process class"""

        for tracked_id in self.tracked_ids_list:

            tracked_id_frames = self._retrieve_frames(tracked_id)
            if tracked_id_frames:
                mop_to_search = self.core_dataframe[
                    self.core_dataframe[TRACKED_ID_COLUMN] == tracked_id
                ][MOP_COLUMN].iloc[0]
                possible_matchs = self.possible_tracked_ids(mop=mop_to_search)

                possible_tracked_id = possible_matchs

                for suspected_match_tracked_id in possible_tracked_id:

                    suspected_match_frames = self._retrieve_frames(
                        suspected_match_tracked_id
                    )

                    non_common_frames = suspected_match_frames.symmetric_difference(
                        tracked_id_frames
                    )

                    common_frames = suspected_match_frames.intersection(
                        tracked_id_frames
                    )

                    if non_common_frames and not common_frames:

                        tracked_id_frame, possible_frame = self._minor_difference(
                            tracked_id_frames, suspected_match_frames, non_common_frames
                        )

                        tracked_id_row = self.retrieve_tracked_id_row(
                            self.tracked_predictions_dataframe,
                            tracked_id,
                            tracked_id_frame,
                        )

                        if not tracked_id_row.empty:

                            suspected_row = self._filter_dataframe(
                                self.tracked_predictions_dataframe,
                                tracked_id_row,
                                possible_frame,
                            )

                            is_same_object = self._is_same_object(
                                tracked_id_row, suspected_row
                            )

                            if is_same_object:

                                self.tracked_predictions_dataframe = (
                                    self._update_dataframe(
                                        self.tracked_predictions_dataframe,
                                        suspected_match_tracked_id,
                                        tracked_id,
                                    )
                                )
                                self.core_dataframe = self._update_dataframe(
                                    self.core_dataframe,
                                    suspected_match_tracked_id,
                                    tracked_id,
                                )

        self.core_dataframe_revised = self.core_dataframe

        return self.core_dataframe_revised


if __name__ == "__main__":
    """
    To test the class:
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--tracked_predictions", type=str, required=True)
    parser.add_argument("--core_report", type=str, required=True)
    parser.add_argument("--new_core_report", type=str, required=False)

    args = parser.parse_args()

    tracked_predictions_dataframe = pd.read_csv(args.tracked_predictions)
    core_dataframe = pd.read_csv(args.core_report)

    service = TrackFixer(tracked_predictions_dataframe, core_dataframe)

    revised_core_dataframe = service.main()

    revised_core_dataframe.to_csv(args.new_core_report)
