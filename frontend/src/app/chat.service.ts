import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { API_URL } from './api.config';

// The shape of the backend's POST /chat response.
export interface ChatResponse {
  answer: string;
  agents_used: string[];
}

@Injectable({ providedIn: 'root' })
export class ChatService {
  private http = inject(HttpClient);

  ask(question: string, role: string, threadId: string): Observable<ChatResponse> {
    return this.http.post<ChatResponse>(`${API_URL}/chat`, {
      question,
      role,
      thread_id: threadId,
    }, {
      headers: { 'ngrok-skip-browser-warning': 'true' },
    });
  }
}
