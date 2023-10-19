using System.Collections;
using System.Collections.Generic;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;
using UnityEngine.AI;

public class NavmeshMovement : MonoBehaviour
{
    //Agent ����
    NavMeshAgent agent;

    //FieldManagement ��ũ��Ʈ���� �ܾ��
    public GameObject[] arrivialPos;
    int arrivalPointNum;

    // �ð����� (������Ҷ� �Ǵ� ������ ��)
    private float timeData;
    int arrivialNum;
    int randomInt;

    //bool arrivalState = false;

    void Start()
    {
        // �ܺ� ���� �ܾ����
        arrivalPointNum = GameObject.Find("SimulationManagement").GetComponent<FieldManagement>().arrivalPointNum;
        arrivialPos = GameObject.Find("SimulationManagement").GetComponent<FieldManagement>().arrivalPointArray;

        // ������Ʈ �ʱ�ȭ
        agent = GetComponent<NavMeshAgent>();

        // ���� �ʱ�ȭ
        arrivialNum = 0;
        timeData = 0;
        randomInt = Random.Range(0, arrivalPointNum);

    }

    void Update()
    {
        // Ÿ��
        agent.SetDestination(arrivialPos[randomInt].transform.position);

        timeData += Time.deltaTime;
        if (timeData > 3) {
            timeData = 0;
            Debug.Log(arrivialPos[randomInt].name);
        }

    }
   


    private void OnTriggerEnter(Collider col)
    {
        if (col.gameObject == arrivialPos[randomInt]) {
            Debug.Log("�浹����");
           
            arrivialNum = randomInt;
            for (;;) {
                randomInt = Random.Range(0, arrivalPointNum);
                if (randomInt != arrivialNum) break;
            
            }
            
           
        }
    }

}
