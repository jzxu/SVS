/* This file is part of the Zenipex Library (zenilib).
 * Copyright (C) 2011 Mitchell Keith Bloch (bazald).
 *
 * zenilib is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * zenilib is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with zenilib.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef ZENI_VECTOR3F_HXX
#define ZENI_VECTOR3F_HXX

#include <Zeni/Vector3f.h>

// HXXed below
#include <Zeni/Coordinate.h>
#include <Zeni/Vector2f.h>

// Not HXXed
#include <cassert>
#include <cmath>

namespace Zeni {

  Vector3f::Vector3f(const bool &degenerate_)
    : i(0.0f), j(0.0f), k(0.0f), degenerate(degenerate_)
  {
  }

  Vector3f::Vector3f(const float &i_, const float &j_, const float &k_, const bool &degenerate_)
    : i(i_), j(j_), k(k_), degenerate(degenerate_)
  {
  }

  Vector3f::Vector3f(const Vector3f &rhs, const bool &degenerate_)
    : i(rhs.i), j(rhs.j), k(rhs.k), degenerate(rhs.degenerate || degenerate_)
  {
  }
  
  Vector3f::Vector3f(const Point3f &rhs)
  : i(rhs.x), j(rhs.y), k(rhs.z)
  {
  }

  Vector3f::Vector3f(const Vector2f &rhs)
  : i(rhs.x), j(rhs.y), k(0.0f)
  {
  }

  Vector3f Vector3f::operator+(const Vector3f &rhs) const {
    return Vector3f(i + rhs.i,
      j + rhs.j,
      k + rhs.k,
      degenerate || rhs.degenerate);
  }

  Vector3f Vector3f::operator-(const Vector3f &rhs) const {
    return Vector3f(i - rhs.i,
      j - rhs.j,
      k - rhs.k,
      degenerate || rhs.degenerate);
  }

  Vector3f & Vector3f::operator+=(const Vector3f &rhs) {
    i += rhs.i;
    j += rhs.j;
    k += rhs.k;
    degenerate |= rhs.degenerate;
    return *this;
  }

  Vector3f & Vector3f::operator-=(const Vector3f &rhs) {
    i -= rhs.i;
    j -= rhs.j;
    k -= rhs.k;
    degenerate |= rhs.degenerate;
    return *this;
  }

  float Vector3f::operator*(const Vector3f &rhs) const {
    return
      i * rhs.i +
      j * rhs.j +
      k * rhs.k;
  }

  Vector3f Vector3f::operator%(const Vector3f &rhs) const {
    return Vector3f(j * rhs.k - rhs.j *k,
      rhs.i *k - i * rhs.k,
      i * rhs.j - rhs.i *j,
      degenerate || rhs.degenerate);
  }

  Vector3f & Vector3f::operator%=(const Vector3f &rhs) {
    degenerate |= rhs.degenerate;
    return *this = *this % rhs;
  }

  Vector3f Vector3f::operator*(const float &rhs) const {
    return Vector3f(i * rhs, j * rhs, k * rhs, degenerate);
  }

  Vector3f Vector3f::operator/(const float &rhs) const {
    return Vector3f(i / rhs, j / rhs, k / rhs, degenerate);
  }

  Vector3f & Vector3f::operator*=(const float &rhs) {
    i *= rhs;
    j *= rhs;
    k *= rhs;
    return *this;
  }

  Vector3f & Vector3f::operator/=(const float &rhs) {
    i /= rhs;
    j /= rhs;
    k /= rhs;
    return *this;
  }

  Vector3f Vector3f::operator-() const {
    return *this * -1;
  }

  // Vector Scalar Multiplication Part II of II
  Vector3f operator*(const float &lhs, const Vector3f &rhs) {
    return rhs * lhs;
  }

  float Vector3f::magnitude2() const {
    return i * i + j * j + k * k;
  }

  float Vector3f::magnitude() const {
    return float(sqrt(magnitude2()));
  }

  Vector3f Vector3f::get_i() const {
    return Vector3f(i, 0.0f, 0.0f);
  }

  Vector3f Vector3f::get_j() const {
    return Vector3f(0.0f, j, 0.0f);
  }

  Vector3f Vector3f::get_k() const {
    return Vector3f(0.0f, 0.0f, k);
  }

  Vector3f Vector3f::get_ij() const {
    return Vector3f(i, j, 0.0f);
  }

  Vector3f Vector3f::get_ik() const {
    return Vector3f(i, 0.0f, k);
  }

  Vector3f Vector3f::get_jk() const {
    return Vector3f(0.0f, j, k);
  }

  Vector3f Vector3f::multiply_by(const Vector3f &rhs) const {
    return Vector3f(i*rhs.i, j*rhs.j, k*rhs.k, degenerate || rhs.degenerate);
  }

  Vector3f Vector3f::divide_by(const Vector3f &rhs) const {
    return Vector3f(i/rhs.i, j/rhs.j, k/rhs.k, degenerate || rhs.degenerate);
  }

  float Vector3f::angle_between(const Vector3f &rhs) const {
    const float a = magnitude();
    const float b = rhs.magnitude();
    const float c = (rhs - *this).magnitude();

    return float(acos((a * a + b * b - c * c) / (2 * a * b)));
  }
  
  const float & Vector3f::operator[](const int &index) const {
    assert(-1 < index && index < 3);
    const float * const ptr = &i;
    return ptr[index];
  }
   
  float & Vector3f::operator[](const int &index) {
    assert(-1 < index && index < 3);
    float * const ptr = &i;
    return ptr[index];
  }

}

#include <Zeni/Coordinate.hxx>
#include <Zeni/Vector2f.hxx>

#endif
