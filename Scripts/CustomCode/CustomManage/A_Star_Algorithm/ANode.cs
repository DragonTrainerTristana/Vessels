using System.Collections;
using System.Collections.Generic;
using UnityEditor.Experimental.GraphView;
using UnityEngine;

public class ANode : MonoBehaviour
{
    public bool isWalkAble;
    public Vector3 worldPos;

    public ANode(bool nWalkable, Vector3 nWorldPos) { 
    
        isWalkAble = nWalkable;
        worldPos = nWorldPos;

    }


}
