import numpy as np

def pose_candidates_from_E(E):
  transform_candidates = []
  ##Note: each candidate in the above list should be a dictionary with keys "T", "R"
  """ YOUR CODE HERE
  """
  R_90 = np.array([[0,-1,0],[1,0,0],[0,0,1]])
  R_90_2 = np.array([[0,1,0],[-1,0,0],[0,0,1]])

  U,_,V = np.linalg.svd(E)
  T1 = U[:,2] / np.linalg.norm(U[:,2])
  T2 = -T1
  R1 = U @ R_90.T @ V
  R2 = U @ R_90_2.T @ V

  # transform_candidates["T"].append(T1)
  # transform_candidates["T"].append(T1)
  # transform_candidates["T"].append(T2)
  # transform_candidates["T"].append(T2)

  transform_candidates.append({"T" : T1 , "R" : R1})
  transform_candidates.append({"T" : T1 , "R" : R2})
  transform_candidates.append({"T" : T2 , "R" : R1})
  transform_candidates.append({"T" : T2 , "R" : R2})

  # transform_candidates[("T", "R")] = [T1,R2]
  # transform_candidates[("T", "R")] = [T2,R1]
  # transform_candidates[("T", "R")] = [T2,R2]

  """ END YOUR CODE
  """
  return transform_candidates