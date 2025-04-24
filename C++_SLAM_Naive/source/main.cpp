//OpenGL
#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <GL/glu.h>
#include <GLFW/glfw3.h>


//OpenCV
#include <opencv2/opencv.hpp>

//g2o
#define G2O_USE_VENDORED_CERES
#include <g2o/core/block_solver.h>
#include  <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/sba/types_six_dof_expmap.h>


//Standard
#include <chrono>
#include <cstdio>
#include <unordered_map>
#include <vector>

class Frame {
    public:
        static inline int id_generator = 0;
        int id;
        cv::Mat image_grey;
        cv::Matx33d K;
        cv::Mat dist;
        std::vector<cv::KeyPoint> kps;
        cv::Mat des;
        cv::Matx33d rotation;
        cv::Matx31d translation;
    
    public:
        Frame() = default;
        Frame(const cv::Mat& input_image_grey, const cv::Matx33d& input_K, const cv::Mat& input_dist) {
            static cv::Ptr<cv::ORB> extractor = cv::ORB::create();
        
            this->id = Frame::id_generator++;
            this->image_grey = input_image_grey;
            this->K = input_K;
            this->dist = input_dist;
        
            std::vector<cv::Point2f> corners;
            cv::goodFeaturesToTrack(this->image_grey, corners, 3000, 0.01, 7);
        
            this->kps.reserve(corners.size());
            for (const cv::Point2f& corner : corners) {
                this->kps.push_back(cv::KeyPoint(corner, 20));
            }
        
            extractor->compute(this->image_grey, this->kps, this->des);
        
            this->rotation = cv::Matx33d::eye();
            this->translation = cv::Matx31d::zeros();
        
            std::printf("Detected: %zu features\n", this->kps.size());
        }
};



class Landmark {
    public:
        static inline int generator =0;
        int id;
        cv::Matx31d location;
        cv::Matx31d colour;
    public:
        Landmark() = default;
        Landmark(const cv::Matx31d& input_location, const cv::Matx31d& input_color){
            this->id = Landmark::id_generator++;
            this->Location = input_location;
            this->colour = input_colour;
        }
};


class Map{
    public:
        std::unordered_map<int, Frame> frames;
        std::unordered_map<int, Landmark> landmarks;
        std::unordered_map<int, std::vector<std::pair<int, int>>> observations; // key is landmark_di, pair are frame_id and kp_index

    public:
        void add_frame(const Frame& frame){
            frames[frame.id] = frame;
        }

        void add_landmark(const Landmark& landmark){
            this->landmarks[landmark.id] = landmark;
        }

        void add_observation(const Frame& frame, const Landmark& landmark, int kp_index){
            this->observations[landmark.id].push_back({frame.id, kp_index});
        }

        static cv::Matx41d triangulate(const Frame& lhs_frame, 
                                       const Frame& rhs_frame,
                                       const cv::Point2d& lhs_point, 
                                       const cv::Point2d& rhs_point){
               
               cv::Matx34d cam1 = {
                lhs_frame.rotation(0, 0), lhs_frame.rotation(0, 1), lhs_frame.rotation(0, 2), lhs_frame.translation(0),
                lhs_frame.rotation(1, 0), lhs_frame.rotation(1, 1), lhs_frame.rotation(1, 2), lhs_frame.translation(1),
                lhs_frame.rotation(2, 0), lhs_frame.rotation(2, 1), lhs_frame.rotation(2, 2), lhs_frame.translation(2),

               };
               cv::Matx34d cam2 = {
                rhs_frame.rotation(0, 0), rhs_frame.rotation(0, 1), rhs_frame.rotation(0, 2), rhs_frame.translation(0),
                rhs_frame.rotation(1, 0), rhs_frame.rotation(1, 1), rhs_frame.rotation(1, 2), rhs_frame.translation(1),
                rhs_frame.rotation(2, 0), rhs_frame.rotation(2, 1), rhs_frame.rotation(2, 2), rhs_Frame.translation(2),
               };
               std::vector<cv::Point2d> lhs_point_normalised;
               cv::undistortPoints(std::vector<cv::Point2d>{lhs_point}, lhs_point_normalised, lhs_frame_K, lhs_frame.dist);
               std::vector<cv::Point2d> rhs_point_normalised;
               cv::undistortPoints(std::vector<cv::Point2d>{rhs_point}, rhs_point_normalised, rhs_frame.K, rhs_frame.dist);
               cv::Mat potential_landmark(4, 1, CV_64F);
               cv::triangulatePoints(cam1, cam2, lhs_point_normalised, rhs_point_normalised, potential_landmark);
               return cv::Matx41d{
                potential_landmark.at<double>(0, 0), // 4x1 matrix acessing  each row at column 0
                potential_landmark.at<double>(1),
                potential_landmark.at<double>(2),
                potential_landmark.at<double>(3)
               };

        } 

        void optimise(int local_window, bool fix_landmarks, int rounds){
            std::unqiue_ptr<g2o::BlockSolver_6_3::LinearSolverType> linear_solver = std::make_unique<g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>>();
            std::unqiue_ptr<g2o::BlockSolver_6_3> solver_ptr = std::make_unique<g2o::BlockSolver_6_3>(std::move(linear_solver));
            g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(std::move(solver_ptr));
            solver->setWriteDebug(false);
            g2o::SparseOptimizer opt;
            opt.setAlgorithm(solver);
            //Add frames
            int non_fixed_poses = 0;
            int landmark_id_start = 0;
            const int local_window_below = Frame::id_generator - 1 - local_window;
            const int local_window_fixed_below = local_window_below + 1 + (fix_landmarks == false);
            for (const auto& [frame_id, frame] : this->frames){
                if ((local_window > 0) && (frame_id < local_window_below)){
                    continue;
                }
                const g2o::Matrix3 rotation = Eigen::Map<const Eigen::Matrix3d>(frame.rotation.t().val); //Eigen is a column major
                const g2o::Vector3 translation = Eigen::Map<const Eigen::Vector3d>(frame.translation.val);
                g2o::SE3Quat se3(rotation, translation);
                v_se3->setEstimate(se3);
                v_se3->setId(frame_id);
                if ((frame_id <= (fix_landmarks == false)) || ((local_window > 0) && (frame_id < local_window_fixed_below))) {
                    v_se3->setFixed(true);
                }
                else {
                    ++non_fixed_poses;
                }
                opt.addVertex(v_se3);
                if (frame_id > landmark_id_start) {
                    landmark_id_start = frame_id + 1;
                }
            }

            // Add landmarks
            int non_fixed_landmarks = 0;
            int non_fixed_edges = 0;
            for (const auto& [landmark_id, landmark] : this->landmarks){
                bool in_local = false;
                for (const auto& [frane_id, kp_index] : this->observations[landmark_id]){
                    if ((local_window == 0 || (!frame_id < local_widnow_below))){
                        in_local = true;
                        break;
                    }
                }
                if (!in_local){
                    continue;
                }

                const cv::Matx31d& cvpt = this->landmarks.at(landmark_id).location;
                const g2o::Vector3 pt(cvpt(0), cvpt(2));
                g2o::VertexPointXYZ* v_pt = new g2o::VertexPointXYZ();
                v_pt->setId(landmark_id_start + landmark_id);
                v_pt->setEstimate(pt);
                v_pt->setMarginalized(true);
                v_pt->setFixed(fix_landmarks);
                non_fixed_landmarks += (fix_landmarks == false);
                opt.addVertex(v_pt);
                // Add edges
                for (const auto& [frame_id, kp_index]: this->observations[landmark_id]){
                    if ((local_window > 0) && (frame_id < local_window_below)){
                        continue;
                    }
                    const Frame& frame = frames.at(frame_id);
                    const cv::Point2f& cvkp = frame.kps[static_cast<unsigned int>(kp_index)].pt;
                    const cv::Matx33d& cvk = frame.K;
                    g2o::EdgeSE3ProjectXYZ* edge = new g2o::EdgeSE3ProjectXYZ();
                    edge->setVertex(0, opt.vertex(landmark_id_start + landmark_id));
                    edge->setVertex(1, opt.vertex(frame_id));
                    if ((frame_id > (fix_landmarks == false)) && (landmark_id >= 0)){
                        ++non_fixed_edges;
                    }
                    edge->setMeasurement(g2o::Vector2(cvkp.x, cvkp.y));
                    edge->setInformation(g2o::Matrix2::Identity());
                    g2o::RobustKernelHuber* robust_kernel = new g2o::RobustKernelHuber();
                    robust_kernel->setDelta(std::sqrt(5.991));
                    edge->setRobustKernel(robust_kernel);
                    edge->fx  = cvk(0, 0);
                    edge->fy  = cvk(1, 1);
                    edge->cx  = cvk(0, 2);
                    edge->cy  = cvk(1, 2);
                    opt.addEdge(edge)
                }
            }
            //Check for some invalid optimiser states
            if (opt.vertices().empty() || opt.edges().empty()){
                std::printf("Optimised: No edges to optimise [%zu %zu]\n", opt.vertices().size(), opt.edges().size());
                return ;
            }
            if ((non_fixed_poses == 0) || ((fix_landmarks == false) && (non_fixed_landmarks == 0)) || (non_fixed_edges == 0)){
                std::printf("Optimised:No non fixed poses [%d %d %d]\n", non_fixed_poses, non_fixed_landmarks, non_fixed_edges);
                return ;
            }

            // Run the optimisation
            opt.initializeOptimization();
            opt.computeActiveErros();
            const double initialChi2 = opt.activeChi2();
            //opt.setVerbose(true);
            opt.optimize(rounds);
            // Apply optimised vertices to frames and landmarks
            for (const auto& [vertex_id, vertex] : opt.vertices()){
                if (vertex_id < Landmark_id_start){
                    Frame& frame = frames.at(vertex_id);
                    const g2o::VertexSE3Expmap* v_pt = static_cast<const g2o::VertexSE3Expmap*>(vertex);
                    Eigen::Map<Eigen::Matrix3d>(frame.rotation.val) = v_pt->estimate().rotation().matrix().transpose(); // Eigen is column major
                    Eigen::Map<Eigen::Matrix3d>(frame.translation_val) = v_pt->estimate().translation();
                }
                else {
                    Landmark& landmark = this->landmarks.at(vertex_id - landmark_id_start);
                    const g2o::VertexPointXYZ* v_pt = static_cast<const g2o::VertexPointXYZ*>(vertex);
                    Eigen::Map<Eigen::Vector3d>(landmark.location.val) = v_pt.estimate();
                }

            }
            std::printf("Optimised: %f to %f error [%zu %zu]\n", initialChi2, opt.activeChi2(), opt.vertices().size(), opt.edges.size());


        };

        void cull(){
            const size_t landmarks_before_cull = this->landmarks.size();
            for (std::unordered_map<int, Landmark>::iterator it = this->landmarks.begin(); it != this->landmarks.end();){
                const std::pair<int, Landmark>& landmark = *it;
                const std::vector<std::pair<int, int>>& landmark_observations = this->observations.at(landmark.first);
                bool not_seen_in_many_frames = landmarks_observations.size() <= 4;
                bool not_seen_recently = landmark_observations.empty() || ((landmark_observations.back().first + 7) < Frame::id_generator);
                if (not_seen_in_many_frames && not_seen_recently){
                    this->observations.erase(landmark.first);
                    it = this->landmarks.erase(it);
                    continue;
                }
                float reprojection_error = 0.0f;
                for (const auto& [frame_id, kp_index]: landmark_observations){
                    const Frame& frame = this->frames.at(frame_id);
                    const cv::Matx21d measured = {static_cast<double>(frame.kps[static_cast<unsigned int>(kp_index)].pt.x), static_cast<double>(frame.kps[static_cast<unsigned int>(kp_index)].pt.y)};
                    const cv::Matx31d mapped = frame.K * ((frame.rotation * landmark.second.location) + fframe.translation);
                    const cv::Matx21d reprojected{mapped(0) / mapped(2), mapped(1) / mapped(2)};
                    reprojection_error += static_cast<float>(cv::norm(measured - reprojected));
                }
                reprojection_error /= static_cast<float>(landmarks_observations.size());
                if (reprojection_error > 5.991f){
                    this->observations.erase(landmark.first);
                    it = this->landmarks.erase(it);
                    continue;
                }
                ++it;
            }
            const size_t landmarks_after_cull = this->landmarks.size();
            std::printf("Culled: %zu points\n", landmarks_before_cull - landmarks_after_cull);
        }
    };

class SLAM {
    public:
        Map mapp;
    
    public:
        void process_frame(const cv::Matx33d& K, const cv::Mat& image_grey){
            Frame frame(image_grey, K, cv::Mat::zeros(1, 4, CV_64F));
            this->mapp.add_frame(frame);
            // 
            if (frame.id == 0){
                return;
            }
            // Get the most recent pair of frames
            Frame& frame_current = mapp.frames.at(Frame::id_generator - 1);
            Frame& frame_previous = mapp.frames.at(Frame::id_generator -2);
            // Helper functions for filtering matches
            constexpr static const auto ratio_test = [](std::vector<std::vector<cv::DMatch>>& matches){
                matches.erase(std::remove_if(matches.begin(), matches.end(), [](const std::vector<cv::DMatch>& options){
                    return ((options.size()<= 1) || ((options[0].distance /  options[1].distance) > 0.75f));
                }), matches.end());
            };

            constexpr static const auto symmetry_test = [](const std::vector<std::vector<cv::DMatch>> &matches1, const std::vector<std::vector<cv::DMatch>> &matches2, std::vector<std::vecor<cv::DMatch>>& symMatches){
                symMatches.clear();
                for (std::vector<std::vector<cv::DMatch>>::const_iterator matchIterator1 = matches1.begin(); matchIterator1 != matches1.end(); ++matchIterator1){
                    for (std::vector<std::vector<cv::DMatch>>::const_iterator matchIterator2 = matches2.begin(); matchIterator2!= matches2.end();++matchIterator2){
                        if ((*matchIterator1)[0].queryIdx == (*matchIterator2)[0].trainIdx &&(*matchIterator2)[0].queryIdx == (*matchIterator1)[0].trainIdx){
                            symMatches.push_back({cv::DMatch((*matchIterator1)[0].queryIdx, (*matchIterator1)[0].trainIdx, (*matchIterator1)[0].distance)});
                            break;
                        }
                    }
                }
            };

            // Compute matches and filter
            static cv::Ptr<cv::BFMatcher> matcher = cv::BFMatcher::create(cv::NORM_HAMMING);
            std::vector<std::vectr<cv::DMatch>> matches_cp;
            std::vector<std::vector<cv::DMatch>> matches_pc;
            matcher->knnMatch(frame_current.des, frame_previous.des, matches_cp, 2);
            matcher->knnMatch(frame_previous.des, frame_current.des, matches_pc, 2);
            ratio_test(matches_cp);
            ratio_test(matches_pc);
            std::vector<std::vector<cv::DMatch>> matches;
            symmetry_test(matches_cp, matches_pc, matches);
            //Create final arrays of good matches
            std::vector<int> match_point_current;
            std::vector<int> match_point_previous;
            for (const std::vector<cv::DMatch>& match : matches){
                const cv::DMatch& m = match[0];
                match_index_current.push_back(m.queryIdx);
                match_point_current.push_back({static_cast<double>(frame_current.kps[static_cast<unsigned int>(m.queryIdx)].pt.x), 
                                               static_cast<double>(frame_current.kps[static_cast<unsigned int>(m.queryIdx)].pt.y)});
            }

            std::printf("Matched %zu features to previous frame \n", matches.size());
            // Pose estimation of new frame
            if (frame_current.id < 2){
                cv::Mat E;
                cv::Mat R;
                cv::Mat t;
                int inliers = cv::RecoverPose(match_point_current, match_point_previous, frame_current.K, frame_current.dist
                                                frame_previous.K, frame_previous.dist, E, R, t, cv::RANSAC, 0.999, 1.0);
                cv::Matx33d rotation = {
                    R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2),
                    R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2),
                    R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2),
                };

                cv::Matx31d translation = {
                    t.at<double>(0),
                    t.at<double>(1),
                    t.at<double>(2)
                };
                // Recover pose seems to return pose 2 to pose 1 rather than pose 1 to pose 2, so invert it.
                rotation = rotation.t();
                translation = -rotation * translation;
                // Set the initial pose of the new frame
                frame_current.rotation = rotation * frame_previous.rotation;
                frame_current.translation = frame_previous.translation + (frame_previous.rotation * translation);
                std::printf("Inliers: %d inliers in pose estimation\n", inliers);
            }

            else {
                // Set the initial pose of the new frame from the previous frame
                frame_current.rotation  = frame_previous.rotation;
                frame_current.translation = frame_previous.translation;
            }

            // Cache observations from the previous frame that are in this one
            std::unordered_map<int, int> frame_previous_points; // key is kp_index and data ia landmark_id

            for (const auto& [landmark_id, landmark_observations] : this.mapp.observations){
                for (const auto& [frame_id, kp_index] : landmark_observations){
                    if (frame_id == frame_previous.id){
                        frame_previous_points[kp_index] = landmark_id;
                    }
                }
            }
            // Find matches that are already in the map as landmarks
            int observations_of_landmarks = 0;
            for (size_t i = 0; i < match_index_previous.size(); ++i){
                auto found_point = frame_previous_points.find(match_index_previous[i]);
                if (found_point != frame_previous_points.end()){
                    // Add observations to features that match existing landmarks
                    this->mapp.add_observation(frame_current, this->mapp.landmarks[found_point->second], match_index_current[i]);
                    ++observations_of_landmarks; 
                }
            }

            




        }

}