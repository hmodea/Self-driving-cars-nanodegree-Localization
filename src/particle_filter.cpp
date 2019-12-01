/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  
  num_particles = 100;  // TODO: Set the number of particles
  std::normal_distribution<double> x_dist(x, std[0]);
  std::normal_distribution<double> y_dist(y, std[1]);
  std::normal_distribution<double> theta_dist(theta, std[2]);

  std::default_random_engine gen;

  //add noise

  for (int i = 0; i < num_particles; ++i) {
    Particle particle{};

    particle.x = x_dist(gen);
    particle.y = y_dist(gen);
    particle.theta = theta_dist(gen);
    particle.weight = 1.0;
    particle.id = i;

    particles.emplace_back(particle);
    weights.emplace_back(1.0);

  }

  is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
  

  

  std::default_random_engine gen;
  double epsilon{std::numeric_limits<double>::epsilon()};
  double velocity_div_yaw_rate{(velocity / (yaw_rate))};
  double yaw_rate_times_delta_t{yaw_rate * delta_t};

  for (int i = 0; i < particles.size(); ++i) {
    double x{particles[i].x};
    double y{particles[i].y};
    double theta{particles[i].theta};

    if (fabs(yaw_rate) < epsilon) {
      x += velocity * delta_t * cos(theta);
      y += velocity * delta_t * sin(theta);
    } else {
      x += velocity_div_yaw_rate * (sin(theta + yaw_rate_times_delta_t) - sin(theta));
      y += velocity_div_yaw_rate * (cos(theta) - cos(particles[i].theta + yaw_rate_times_delta_t));
      theta += yaw_rate * delta_t;
    }

    //adding gaussian noise
    std::normal_distribution<double> x_dist(x, std_pos[0]);
    std::normal_distribution<double> y_dist(y, std_pos[1]);
    std::normal_distribution<double> theta_dist(theta, std_pos[2]);

    particles[i].x = x_dist(gen);
    particles[i].y = y_dist(gen);
    particles[i].theta = theta_dist(gen);

  }

}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted,
                                     vector<LandmarkObs> &observations) {
  for (auto &observation : observations) {
    double min_distance{std::numeric_limits<double>::max()};

    for (auto &prediction : predicted) {
      double distance{dist(prediction.x, prediction.y, observation.x, observation.y)};

      if (distance < min_distance) {
        observation.id = prediction.id;
        min_distance = distance;
      }
    }
  }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {

  const double var_x{std_landmark[0] * std_landmark[0]};
  const double var_y{std_landmark[1] * std_landmark[1]};
  const double gauss_norm{(1. / (2. * M_PI * std_landmark[0] * std_landmark[1]))};

  for (auto &particle : particles) {
    vector<LandmarkObs> transformed_observations;
    vector<LandmarkObs> landmarks_within_sensor_range;
    particle.weight = 1.0;

    for (const auto &observation : observations) {
      double trns_x = particle.x + (observation.x * cos(particle.theta)) - (observation.y * sin(particle.theta));
      double trns_y = particle.y + (observation.x * sin(particle.theta)) + (observation.y * cos(particle.theta));
      transformed_observations.push_back({observation.id, trns_x, trns_y});
    }

    for (auto &landmark : map_landmarks.landmark_list) {
      double distance{dist(landmark.x_f, landmark.y_f, particle.x, particle.y)};
      if (distance <= sensor_range) {
        landmarks_within_sensor_range.push_back({landmark.id_i, landmark.x_f, landmark.y_f});
      }
    }

    dataAssociation(landmarks_within_sensor_range, transformed_observations);

    for (const auto &transformed_observation : transformed_observations) {
      for (const auto &landmark : landmarks_within_sensor_range) {
        if (transformed_observation.id == landmark.id) {
          const double dx = transformed_observation.x - landmark.x;
          const double dy = transformed_observation.y - landmark.y;
          const double exponent{((dx * dx) / (2. * var_x))+ ((dy * dy) / (2. * var_y))};
          particle.weight *= gauss_norm * exp(-exponent);
          break;
        }
      }
    }
  }
}


void ParticleFilter::resample() {

  for (int i = 0; i < num_particles; ++i) {
    weights[i] = particles[i].weight;
  }

  std::default_random_engine gen;
  std::discrete_distribution<int> importance_dist(weights.begin(), weights.end());
  std::vector<Particle> resampled_particles{};

  for (auto i = 0; i < num_particles; ++i) {
    resampled_particles.emplace_back(particles[importance_dist(gen)]);
  }

  particles = resampled_particles;

}

void ParticleFilter::SetAssociations(Particle &particle,
                                     const vector<int> &associations,
                                     const vector<double> &sense_x,
                                     const vector<double> &sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);// get rid of the trailing space
  return s;
}
