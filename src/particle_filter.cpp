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

static std::default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 20;  // TODO: Set the number of particles

  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);

  for(unsigned int i = 0; i < num_particles; i++)
  {
    Particle particle;
    particle.id = i;
    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
    particle.weight = 1.0;
    // weights[i] = 1.0;

    particles.push_back(particle);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */

  // std::default_random_engine gen;

  for(unsigned int i = 0; i < num_particles; ++i)
  {
    if(fabs(yaw_rate) < 0.00001)
    {
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t *sin(particles[i].theta);
    }

    else
    {
      particles[i].x += (velocity / yaw_rate) * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
      particles[i].y += (velocity / yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
      particles[i].theta += yaw_rate * delta_t;
    }

    std::normal_distribution<double> dist_x(0, std_pos[0]);
    std::normal_distribution<double> dist_y(0, std_pos[1]);
    std::normal_distribution<double> dist_theta(0, std_pos[2]);

    particles[i].x += dist_x(gen);
    particles[i].y += dist_y(gen);
    particles[i].theta += dist_theta(gen);    
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */

  for(unsigned int i = 0; i < observations.size(); i++)
  {
    double error = std::numeric_limits<double>::max();
    double distance;
    int idx = -1;

    for(unsigned int j = 0; j < predicted.size(); ++j)
    {
      distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);

      if(distance < error)
      {
        error = distance;
        idx = predicted[j].id;
      }

      observations[i].id = idx;
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

  for(unsigned int i = 0; i < num_particles; i++)
  {
    // Weight reinitiation
    particles[i].weight = 1.0;

    // Landmark filter based on the sensor range
    vector<LandmarkObs> filtered_landmarks;

    for(unsigned int k = 0; k < map_landmarks.landmark_list.size(); k++)
    {
      double dis = dist(particles[i].x, particles[i].y, map_landmarks.landmark_list[k].x_f, map_landmarks.landmark_list[k].y_f);

      if(dis <= sensor_range)
      {
        LandmarkObs filtered_landmark;
        filtered_landmark.id = map_landmarks.landmark_list[k].id_i;
        filtered_landmark.x = map_landmarks.landmark_list[k].x_f;
        filtered_landmark.y = map_landmarks.landmark_list[k].y_f;

        filtered_landmarks.push_back(filtered_landmark);
      }
    }

    vector<LandmarkObs> mapped_observations;

    for(unsigned int j = 0; j < observations.size(); j++)
    {
      LandmarkObs mapped_observation;

      mapped_observation.id = observations[j].id;
      mapped_observation.x = particles[i].x + (cos(particles[i].theta)*observations[j].x) - (sin(particles[i].theta)*observations[j].y);
      mapped_observation.y = particles[i].y + (sin(particles[i].theta)*observations[j].x) + (cos(particles[i].theta)*observations[j].y);
      mapped_observations.push_back(mapped_observation);
    }

    // DataAssociation
    dataAssociation(filtered_landmarks, mapped_observations);

    // Weight Update
    for(int l = 0; l < mapped_observations.size(); l++)
    {
      Map::single_landmark_s map_landmark = map_landmarks.landmark_list.at(mapped_observations[l].id-1);

      double diff_x = mapped_observations[l].x - map_landmark.x_f;
      double diff_y = mapped_observations[l].y - map_landmark.y_f;

      double sigma_x = std_landmark[0];
      double sigma_y = std_landmark[1];
      double norm = (2.0*M_PI*sigma_x*sigma_y);

      double guassian_dist = (1/norm)*exp(-(((pow(diff_x,2))/(2*pow(sigma_x,2)))+((pow(diff_y,2))/(2*pow(sigma_y,2)))));
      particles[i].weight *=  guassian_dist;
    }
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  std::random_device rand;
  std::mt19937 gen2(rand());
   vector<double> particle_weights;

   for (int i = 0; i < particles.size(); i++)
   {
     particle_weights.push_back(particles[i].weight);
   }

  std::discrete_distribution<> weight_dist(particle_weights.begin(), particle_weights.end());
  
  vector<Particle> resampled_particles;

  int idx;
  
  for(int i = 0; i < num_particles; i++)
  {
      idx = weight_dist(gen2);
      resampled_particles.push_back(particles[idx]);
  }

  particles = resampled_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
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
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}