using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

public class CustomAgentMovement : MonoBehaviour
{

    //RayCast & LineRenderer
    public float rayLength = 5.0f;
    public LineRenderer lineRenderer;
    public int rayCount = 36;
    public float raySpreadAngle = 10f;
    public float[] rayObjDis;

    // ������
    public Transform myWayPoint;
    public float distance;

     // ������Ʈ ������Ʈ ����
   [SerializeField]
    Rigidbody rb;
    public bool isAlive;

    // �� ������
    public float moveSpeed = 0.1f;
    public float rotationSpeed = 0.1f;
    

    void Start()
    {
        // �Ÿ� ���
        distance = Vector3.Magnitude(this.gameObject.transform.position - myWayPoint.position);

        // Rigidbody �ʱ�ȭ
        rb = GetComponent<Rigidbody>();
        rb.velocity = Vector3.zero;
        rb.angularVelocity = Vector3.zero;

        // RayCast & LineRenderer ������Ʈ ��������
        rayObjDis = new float[36];
        for (int i = 0; i < 36; i++) rayObjDis[i] = rayLength;

        lineRenderer = gameObject.AddComponent<LineRenderer>();
        lineRenderer.positionCount = rayCount * 2; // �������� ������ ���� �ʿ��ϹǷ� *2
        lineRenderer.startWidth = 0.01f;
        lineRenderer.endWidth = 0.01f;
        lineRenderer.material = new Material(Shader.Find("Sprites/Default"));
        lineRenderer.startColor = Color.red;
        lineRenderer.endColor = Color.red;

        // ���º���
        isAlive = true;
    }

    void Update()
    {
        // ���� �׽�Ʈ�� GetKeyDown (w,a,d�θ� �غ���, ���� DDPG������ a,d �� Rudder�� ��������)
        // �׸��� Agent Drag coefficient ��� �־����

        //Ray
        performRayCast();
        //Movement
        vesselMovement();
    }

    //override void OnEpisodeBegin()
    //{

    //}

    public void vesselMovement()
    {
        float horizontalInput = Input.GetAxis("Horizontal");
        float verticalInput = Input.GetAxis("Vertical");

        //Vector3 moveDirection = transform.forward * verticalInput;
        //Vector3 rotation = new Vector3(0, horizontalInput, 0) * rotationSpeed * Time.deltaTime;

        //rb.AddForce(moveDirection * moveSpeed);
        //rb.AddTorque(rotation);

        Vector3 moveDirection = transform.forward * verticalInput;
        Vector3 velocity = moveDirection * moveSpeed;

        // ���� �ӵ��� ��ǥ �ӵ� ���̸� �����Ͽ� �ε巯�� �̵� ����
        rb.velocity = Vector3.Lerp(rb.velocity, velocity, 0.1f);

        // �ε巯�� ȸ�� ����
        float turn = horizontalInput * rotationSpeed * Time.deltaTime;
        Quaternion turnRotation = Quaternion.Euler(0f, turn, 0f);
        rb.MoveRotation(rb.rotation * turnRotation);

    }

    public void performRayCast()
    {
        for (int i = 0; i < rayCount; i++)
        {
            float angle = i * raySpreadAngle - (raySpreadAngle * (rayCount - 1) / 2);
            Quaternion rotation = Quaternion.Euler(0f, angle, 0f);
            Vector3 direction = rotation * transform.forward;

            Ray ray = new Ray(transform.position, direction);
            RaycastHit hit;

            int index = i * 2; // ���̴� �������� ���� 2���� �������ϱ� ���� *2
            lineRenderer.SetPosition(index, ray.origin);

            if (Physics.Raycast(ray, out hit, rayLength))
            {
                // ���̰� �浹�� ���
                lineRenderer.SetPosition(index + 1, hit.point);
                rayObjDis[i] = hit.distance;

            }
            else
            {
                // ���̰� �浹���� ���� ���
                lineRenderer.SetPosition(index + 1, ray.origin + ray.direction * rayLength);
                rayObjDis[i] = rayLength;

            }
        }
    }

    void OnTriggerEnter(Collider other)
    {
        if (other.CompareTag("Obstacle")) // Vessel Agent�̵�, ���̵簣�� ���� tag Obstacle ó��
        {
            isAlive = false;

        }

    }


}
