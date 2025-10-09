import torch

def flash_attention_forward(Q, K, V, B_q=2, B_k=2):
    """
    FlashAttention forward pass
    
    Q: (B, N, d)  Queries
    K: (B, N, d)  Keys
    V: (B, N, d_v) Values
    
    Returns:
        O: (B, N, d_v)
    """
    B, N, d = Q.shape
    d_v = V.shape[-1]
    O = torch.zeros(B, N, d_v)

    for b in range(B):  # batch
        for i in range(0, N, B_q):  # query blocks
            Q_i = Q[b, i:i+B_q]  # (B_q, d)

            # init accumulators (paper notation)
            m_i = torch.full((Q_i.shape[0],), -float("inf"))  # running max
            l_i = torch.zeros(Q_i.shape[0])                   # running denominator
            acc_i = torch.zeros(Q_i.shape[0], d_v)            # running numerator

            for j in range(0, N, B_k):  # key blocks
                K_j = K[b, j:j+B_k]  # (B_k, d)
                V_j = V[b, j:j+B_k]  # (B_k, d_v)

                # (1) Scores
                S_ij = (Q_i @ K_j.T) / (d ** 0.5)  # (B_q, B_k)

                # (2) Row max
                m_ij = S_ij.max(dim=1).values

                # (3) Exp-normalized scores
                P_ij = torch.exp(S_ij - m_ij[:, None])

                # (4) Partial sums
                l_ij = P_ij.sum(dim=1)         # (B_q,)
                acc_ij = P_ij @ V_j            # (B_q, d_v)

                # (5) Merge with running totals
                m_new = torch.maximum(m_i, m_ij)
                l_i = l_i * torch.exp(m_i - m_new) + l_ij * torch.exp(m_ij - m_new)
                acc_i = acc_i * torch.exp((m_i - m_new)[:, None]) + acc_ij * torch.exp((m_ij - m_new)[:, None])
                m_i = m_new

            # Final normalize
            O[b, i:i+B_q] = acc_i / l_i[:, None]

    return O

if __name__ == "__main__":
    # Simple test
    torch.manual_seed(0)
    B, N, d, d_v = 1, 4, 2, 2
    Q = torch.randn(B, N, d)
    K = torch.randn(B, N, d)
    V = torch.randn(B, N, d_v)

    out_flash = flash_attention_forward(Q, K, V, B_q=2, B_k=2)

    # Compare with naive softmax attention
    def naive_attention(Q, K, V):
        S = Q @ K.transpose(-2, -1) / (Q.shape[-1] ** 0.5)
        P = torch.softmax(S, dim=-1)
        return P @ V

    out_naive = naive_attention(Q, K, V)

    print("FlashAttention (paper notation):\n", out_flash)
    print("\nNaive attention:\n", out_naive)
    
