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
        };