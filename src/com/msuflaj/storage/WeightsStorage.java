package com.msuflaj.storage;

import com.msuflaj.network.Network;

public interface WeightsStorage {

    public boolean store(Network net);

    public boolean load(Network net);

}
