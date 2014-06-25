#include "change_tracking_list.h"

template<class T>
change_tracking_list<T>::change_tracking_list() {
	m_added_begin = 0;
}

template<class T>
change_tracking_list<T>::~change_tracking_list() {
  clear();
}

template<class T>
void change_tracking_list<T>::add(T* v){ 
	current.push_back(v);
	for (int i = 0; i < listeners.size(); ++i) {
		listeners[i]->handle_ctlist_add(v);
	}
}

template<class T>
void change_tracking_list<T>::remove(const T* v) {
	bool found = false;
	for (int i = 0; i < current.size(); ++i) {
		if (current[i] == v) {
			removed.push_back(current[i]);
			current.erase(current.begin() + i);
			if (i < m_added_begin) {
				--m_added_begin;
			}
			found = true;
			break;
		}
	}
	assert(found);
	for (int i = 0; i < changed.size(); ++i) {
		if (changed[i] == v) {
			changed.erase(changed.begin() + i);
			break;
		}
	}
	for (int i = 0; i < listeners.size(); ++i) {
		listeners[i]->handle_ctlist_remove(v);
	}
}

template<class T>
void change_tracking_list<T>::change(const T *v) {
	for(int i = 0; i < current.size(); ++i) {
		if (current[i] == v) {
			if (i < m_added_begin &&
					find(changed.begin(), changed.end(), current[i]) == changed.end())
			{
				changed.push_back(current[i]);
				for (int i = 0; i < listeners.size(); ++i) {
					listeners[i]->handle_ctlist_change(current[i]);
				}
			}
			return;
		}
	}
	assert(false);
}

template<class T>
void change_tracking_list<T>::clear_changes() {
	m_added_begin = current.size();
	changed.clear();
	clear_removed();
}

// Causes everything to be marked as added
template<class T>
void change_tracking_list<T>::reset() {
	changed.clear();
	clear_removed();
	m_added_begin = 0;
}

// Removes all items from all lists
template<class T>
void change_tracking_list<T>::clear() {
	// Clear changed list
	changed.clear();

	// Clear current list
	m_added_begin = 0;
	for(int i = 0; i < current.size(); i++){
		for (int j = 0; j < listeners.size(); ++j) {
			listeners[j]->handle_ctlist_remove(current[i]);
		}
		removed.push_back(current[i]);
	}
	current.clear();

	// Clear removed list
	clear_removed();
}
	
template<class T>
void change_tracking_list<T>::listen(ctlist_listener<T> *l) {
	listeners.push_back(l);
}

template<class T>
void change_tracking_list<T>::unlisten(ctlist_listener<T> *l) {
	typename std::vector<ctlist_listener<T>*>::iterator i;
	i = std::find(listeners.begin(), listeners.end(), l);
	if (i != listeners.end()) {
		listeners.erase(i);
	}
}

template<class T>
void change_tracking_list<T>::clear_removed() {
	for (int i = 0; i < removed.size(); ++i) {
		delete removed[i];
	}
	removed.clear();
}

