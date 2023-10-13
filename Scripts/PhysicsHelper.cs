using UnityEngine;

namespace Ditzelgames
{
    public static class PhysicsHelper
    {

        // -------------------------- 2023 - 09 - 17 기준 개발기록 정리 --------------------------
        /*
        
        1) 기본적인 물리 계산 스크립트

        생각해보면, Rigidbody라는 컴포넌트를 자세히 봐야할 필요성이 있다.
        어쨌든 유니티는 Physics Simulation platform이고, Realstic한 물리효과가 구현되어 있잖아? (근사 방법)
        '강체', rigidbody component는 Mass / Drag(공기 저항과 Ocean Current에 대한 저항) / Angular Drag (오브젝트가 토크로 회전할 때, 각 저항이 영향을 미치는 정도)

        어쨌든, 각 ocean current의 속도와 각도에 따라 drag magnitude를 스케일링 해줄 필요성이 있다는 것.

        -> Ship Velocity를 V_s로 정의하고, Theta_s가 Ship angle(direction)으로 정의하자.
           그러면 Ocean Current는 V_c, theta_c가 될거고, 결과론적으로 velocity는 V_r로 새로 정의하고 싶은 거다.
        -> V_r은 sqrt(V_s^2 + V_c^2 + 2*|V_s||V_c|cos(theta_s - theta_c))가 될거고, 
        -> theta_r은 arctan( (|V_s|sin(theta_s) + |V_c|sin(theta_c)) / (|V_s|cos(theta_s) + |V_c|cos(theta_c)))가 될 것이다.

        // References
        A. <https://docs.unity3d.com/kr/2018.4/Manual/class-Rigidbody.html -> Rigidbody에 대한 유니티 공식 문서>

        B. <https://api.unity.com/v1/oauth2/authorize?client_id=unity_forum&response_type=code&redirect_uri=https%3A%2F%2Fforum.unity.com
        %2Fregister%2Fgenesis&state=fxHoyeDTVuU59S6pZpA1T6xPL1A84ytirlKL2egE%3B%2Fthreads%2Fdrag-factor-what-is-it.85504%2F&prompt=NONE>
        -> 실제 physcis랑 drag calculation 방식 비교
        a) dragForceMagnitude = velocity.magnitude^2 * drag;  // 실제 Physics
        b) velocty = velocty * (1 - deltaTime * drag) // linear approximation in Unity
        -> 따라서, 유니티에서는 newVelocity = (currentVelocity + forces) * (1 - drag)가 되고 newPosition = currentPosition + currentVelocty가 된다.
        
         */


        // 속도 계산 함수
        public static void ApplyForceToReachVelocity(Rigidbody rigidbody, Vector3 velocity, float force = 1, ForceMode mode = ForceMode.Force)
        {
            if (force == 0 || velocity.magnitude == 0) // 엔진 시동 안걸면, 저항 계산 안 할거다.
                return;

            velocity = velocity + velocity.normalized * 0.2f * rigidbody.drag;

            //force = 1 => need 1 s to reach velocity (if mass is 1) => force can be max 1 / Time.fixedDeltaTime
            force = Mathf.Clamp(force, -rigidbody.mass / Time.fixedDeltaTime, rigidbody.mass / Time.fixedDeltaTime);

            //dot product is a projection from rhs to lhs with a length of result / lhs.magnitude https://www.youtube.com/watch?v=h0NJK4mEIJU
            if (rigidbody.velocity.magnitude == 0)
            {
                rigidbody.AddForce(velocity * force, mode);
            }
            else
            {
                var velocityProjectedToTarget = (velocity.normalized * Vector3.Dot(velocity, rigidbody.velocity) / velocity.magnitude);
                rigidbody.AddForce((velocity - velocityProjectedToTarget) * force, mode);
            }
        }

        // 회전 계산 함수
        public static void ApplyTorqueToReachRPS(Rigidbody rigidbody, Quaternion rotation, float rps, float force = 1)
        {
            var radPerSecond = rps * 2 * Mathf.PI + rigidbody.angularDrag * 20;

            float angleInDegrees;
            Vector3 rotationAxis;
            rotation.ToAngleAxis(out angleInDegrees, out rotationAxis);

            if (force == 0 || rotationAxis == Vector3.zero)
                return;

            rigidbody.maxAngularVelocity = Mathf.Max(rigidbody.maxAngularVelocity, radPerSecond);

            force = Mathf.Clamp(force, -rigidbody.mass * 2 * Mathf.PI / Time.fixedDeltaTime, rigidbody.mass * 2 * Mathf.PI / Time.fixedDeltaTime);

            var currentSpeed = Vector3.Project(rigidbody.angularVelocity, rotationAxis).magnitude;

            rigidbody.AddTorque(rotationAxis * (radPerSecond - currentSpeed) * force);
        }

        public static Vector3 QuaternionToAngularVelocity(Quaternion rotation)
        {
            float angleInDegrees;
            Vector3 rotationAxis;
            rotation.ToAngleAxis(out angleInDegrees, out rotationAxis);

            return rotationAxis * angleInDegrees * Mathf.Deg2Rad;
        }

        public static Quaternion AngularVelocityToQuaternion(Vector3 angularVelocity)
        {
            var rotationAxis = (angularVelocity * Mathf.Rad2Deg).normalized;
            float angleInDegrees = (angularVelocity * Mathf.Rad2Deg).magnitude;

            return Quaternion.AngleAxis(angleInDegrees, rotationAxis);
        }

        public static Vector3 GetNormal(Vector3[] points)
        {
            //https://www.ilikebigbits.com/2015_03_04_plane_from_points.html
            if (points.Length < 3)
                return Vector3.up;

            var center = GetCenter(points);

            float xx = 0f, xy = 0f, xz = 0f, yy = 0f, yz = 0f, zz = 0f;

            for (int i = 0; i < points.Length; i++)
            {
                var r = points[i] - center;
                xx += r.x * r.x;
                xy += r.x * r.y;
                xz += r.x * r.z;
                yy += r.y * r.y;
                yz += r.y * r.z;
                zz += r.z * r.z;
            }

            var det_x = yy * zz - yz * yz;
            var det_y = xx * zz - xz * xz;
            var det_z = xx * yy - xy * xy;

            if (det_x > det_y && det_x > det_z)
                return new Vector3(det_x, xz * yz - xy * zz, xy * yz - xz * yy).normalized;
            if (det_y > det_z)
                return new Vector3(xz * yz - xy * zz, det_y, xy * xz - yz * xx).normalized;
            else
                return new Vector3(xy * yz - xz * yy, xy * xz - yz * xx, det_z).normalized;

        }

        public static Vector3 GetCenter(Vector3[] points)
        {
            var center = Vector3.zero;
            for (int i = 0; i < points.Length; i++)
                center += points[i] / points.Length;
            return center;
        }
    }
}
