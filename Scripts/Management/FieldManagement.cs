using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FieldManagement : MonoBehaviour
{
    // 2023 - 10 -13 ���� ���߱��
    /*
     1) FieldManagement ��ũ��Ʈ���� ��� ������ �ð��� �����ؾ���
         
    
     2) �ϴ� ���� ����ȭ ����̶� �� ������Ѻ��鼭,  
     
     
     */

    // Agent
    public GameObject prefabAgent;
    public GameObject[] agentsArray;
    public int agentNum;

    // Ocean_current data (csvRead �Լ� Ȱ��)
    // ���� ��ǥ -> ����Ƽ ��ǥ scaling �۾� �ʼ�
    
    bool interactionState = true; // agent script ���� state
    
    //  



    // Start is called before the first frame update
    void Start()
    {
        agentsArray = new GameObject[agentNum];


        
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    // extract data
    void csvRead()
    {


    }

}
