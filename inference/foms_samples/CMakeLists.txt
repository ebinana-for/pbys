

if (TARGET_CPU MATCHES "arm")
    add_subdirectory(samples_utils)
    add_subdirectory(samples_common_device)
    add_subdirectory(samples_age_gender_with_csv_and_video_device)
    add_subdirectory(samples_positions_with_csv_and_video_device)

else()
    add_subdirectory(samples_utils)
    add_subdirectory(samples_common)
    add_subdirectory(samples_age_gender)
    add_subdirectory(samples_age_gender_with_csv_and_video)
 
    add_subdirectory(samples_positions_with_csv_and_video)
    add_subdirectory(samples_age_gender_evaluation)
    add_subdirectory(samples_positions_evaluation)
    add_subdirectory(samples_common_device)
    add_subdirectory(samples_age_gender_with_csv_and_video_device)
    add_subdirectory(samples_positions_with_csv_and_video_device)
endif()